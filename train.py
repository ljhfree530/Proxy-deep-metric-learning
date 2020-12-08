# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from pytorch_metric_learning import losses
import numpy as np
import random
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib
from imageloader import image_loader
from test_embedded import Get_test_results_single
from test_embedded import Get_test_results_inshop
from test_embedded import extract_feature
from test_embedded import get_id
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
from tools import get_feature_infos
from tools import softmax_new
from tools import proxy_anchor
from tools import angle_hists
from tools import print_list_values
from tools import stretch_weighted
from tools import get_intra_inter
from tools import get_stretchedness
from tools import get_stretchedness_center
from tools import set_proxies_globalcenter
from tools import initial_proxies
from tools import stretching_proxies
from PIL import Image
from util.loss import normalize
from util.loss import potential
from util.loss import potential_normalized
from util.loss import potential_normalized_softmax
from util.loss import smooth_max
from util.loss import CompactnessLoss
from util.loss import EntropyLoss
from util.loss import HLoss
from util.loss import HLoss_normalized
from util.loss import pairwise_similarity
# from util.loss import TripletLoss_new
from util.loss import pairwise_innerprod
import time
import os
from model import ft_net, ft_net_scratch, ft_net_feature, ft_net_feature_linear_added, ft_net_feature_linear_added_bn
from model import bn_inception_PA
# from tensorboard_logger import configure, log_value
import json
#import visdom
import copy
import PIL
import sklearn.preprocessing
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description=
    'Official implementation of `Proxy Centering for Deep Metric Learning`'
    + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
)

parser.add_argument('--dataset', default='CUB',
    dest = 'data_type',
    help = 'Training dataset, e.g. CUB, CAR, SOP, Inshop'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'feature_size',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 40, type = int,
    dest = 'train_batchsize',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = -1, type = int,
    dest = 'e_end',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    dest = 'gpu_id',
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--model', default = 'RES50',
    dest = 'model_name',
    help = 'Model for training, e.g. RES50, BN'
)
parser.add_argument('--loss', default = 'softmax',
    dest = 'loss_type',
    help = 'Criterion for training'
)
parser.add_argument('--optimizer', default = 'SGD',
    dest = 'optimizer',
    help = 'Optimization Algorithm'
)
parser.add_argument('--initial', default = 'random',
    dest = 'weight_initialization',
    help = 'Criterion for training'
)
parser.add_argument('--model-lr', default = 1e-3, type =float,
    dest = 'model_learning_rate',
    help = 'Learning rate setting'
)
parser.add_argument('--proxy-lr', default = 1e-3, type =float,
    dest = 'classifier_learning_rate',
    help = 'Learning rate setting'
)
parser.add_argument('--gamma', default = 1e-3, type = float,
    dest = 'gamma',
    help = 'Scaling Parameter setting'
)
parser.add_argument('--warm', default = 5, type = int,
    dest = 'warm_epoch',
    help = 'Warmup training epochs'
)

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)


#class lists
####################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(dir_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        # network.cuda()
        network.cuda(gpu_ids[0])
        # nn.DataParallel(network, device_ids=[2,3]).cuda()


class experiment_info():
    def __init__(self, data_type, image_datasets, dataloaders, model_name, class_nb, original_size, feature_size, computer):
        self.data_type = data_type
        self.image_datasets = image_datasets
        self.dataloaders = dataloaders
        self.model_name = model_name
        self.class_nb = class_nb

        self.original_size = original_size
        self.feature_size = feature_size

        self.computer = computer


##################
#
#  Model defining

def load_network_path(network, save_path):
    state_dict_ = torch.load(save_path)
    network.load_state_dict(state_dict_['model_state_dict'])
    print("Loaded")
    return network


lr1 = 0.1
lr2 = 0.01
lr3 = 0.001
lr4 = 0.0001


####################################################
# Set arguments, hyperparameters
#
sds = False
computer = '58'
gpu_id = args.gpu_id
data_type = args.data_type

# model setting
model_name = args.model_name
stride = 1
linear_added = True
feature_size = args.feature_size
# print(model_arg_dict)

# training setting
pretrained = True
set_epoch = 0
if not pretrained:
    set_epoch = -1


optimizer = args.optimizer
model_learning_rate = args.model_learning_rate
layer_learning_rate = 0.0
classifier_learning_rate = args.classifier_learning_rate
#weight initialization: random, fcmean, centerspread, or centerprop
weight_initialization = args.weight_initialization
gamma = args.gamma
# print(train_arg_dict)

# base loss setting
loss_type = args.loss_type

print_epoch_interval = 5

upper_cos = 1.0
lower_cos = 0.8
nb_interval = 20.0


####################################################
#  File/model directory
#

if sds:
    if data_type == 'CUB':
        data_dir = '/home/sds/FG/CUB_RT/pytorch/pytorch'
    elif data_type == 'CAR':
        data_dir = '/home/sds/FG/STCAR_RT/pytorch'
    elif data_type == 'SOP':
        data_dir = '/home/sds/FG/Stanford_Online_Products/pytorch'
    elif data_type == 'Inshop':
        data_dir = '/home/ro/FG/Inshop/pytorch'
else:
    if data_type == 'CUB':
        data_dir = '/home/ro/FG/CUB_RT/pytorch'
    elif data_type == 'CAR':
        data_dir = '/home/ro/FG/STCAR_RT/pytorch'
    elif data_type == 'SOP':
        data_dir = '/home/ro/FG/Stanford_Online_Products/pytorch'
    elif data_type == 'Inshop':
        data_dir = '/home/ro/FG/Inshop/pytorch'

# data_dir = '/data1/home/jhlee/CUB_200_2011_1/pytorch'

# dir_name = '/data/ymro/AAAI2021/base_CUB/res50_lr3_fclr2'
# dir_name = '/data/ymro/AAAI2021/jhlee/CUBlabNet/potential_loss_on'
if sds:
    dir_name = '/home/sds/Dropbox/jhlee'
else:
    dir_name = '/home/ro/Dropbox/jhlee'





######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
# def load_network_path(network, save_path):
#     network.load_state_dict(torch.load(save_path))
#     print("Loaded")
#     return network

# if not os.path.isdir(dir_name):
#     os.mkdir(dir_name)
#
# # configure(dir_name)
# # print(dir_name)
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)

#######################################################
#
# Settings: GPU id, parameters and options.
#

# GPU
use_gpu = True
gpu_ids = []
gpu_ids.append(gpu_id)
# set gpu id2
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
# print(gpu_ids[0])

# epoch_
if data_type == 'Inshop':
    e_drop = 60
    e_drop2 = 0
    e_end = 80
    print_epoch_per = 10
elif data_type == 'SOP':
    e_drop = 60
    e_drop2 = 0
    e_end = 80
    print_epoch_per = 10
elif data_type == 'CAR':
    e_drop = 10
    e_drop2 = 0
    e_end = 20
    print_epoch_per = 1
else:
    e_drop = 10
    e_drop2 = 0
    e_end = 20
    print_epoch_per = 1



# Learning rates
prep_batchsize = 20
train_batchsize = args.train_batchsize
test_batchsize = 20

# Model and loss control
# Model name: RES50, BN

if model_name == 'RES50':
    original_size = 2048
elif model_name == 'BN':
    original_size = 1024
#
#
# print("The LR of linear layer is assigned, and SGD. Layer lr = 5lr4")
if not linear_added:
    feature_size = original_size


######################################################################
# Load Data
# ---------
#

image_datasets, dataloaders = image_loader(model_name, data_type, data_dir, prep_batchsize, train_batchsize, test_batchsize)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes
class_nb = len(class_names)

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))


def train_model(model, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # normalizing proxies
    wt = model.classifier.add_block[0].weight.data
    model.classifier.add_block[0].weight.data = normalize(wt)

    # extract features before training
    features_after = get_feature_infos(model, expInfo, normalized_mean=False, get_more_infos=True)
    before_train_features = features_after['feature list']


    # check the state of pre-trained model
    for phase in ['train']:
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        iter_ = 0

        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels = data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            iter_ += 1

            # forward
            f, outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            # Softmax loss
            loss = 0.0
            if loss_type == 'softmax':
                loss_1 = softmax_new(expInfo, outputs, labels, mrg=1.0)
                loss += loss_1
            elif loss_type == 'PA':
                loss_PA = proxy_anchor(expInfo, outputs, labels)
                loss += loss_PA
            elif loss_type == 'largemargin':
                loss_1 = criterion(f, labels)
                loss += loss_1

            running_loss += loss.data
            running_corrects += torch.sum(preds == labels.data)

        running_corrects = running_corrects.float()
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

    d_intra = features_after['intra']
    d_inter = features_after['inter']
    phi = d_inter / d_intra

    results, intra_test, inter_test = Get_test_results_single(model, expInfo)
    phi_test = inter_test / intra_test

    print('X {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.6f} {:.6f} {:.4f} {:.6f} {:.6f} {:.4f}'.format(epoch_acc, epoch_loss, results[0], results[1], results[2], results[3], d_intra, d_inter, phi, intra_test, inter_test, phi_test))

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase

        if epoch == set_epoch:
            # if weight_initialization != 'random':
            feature_mean = Variable(features_after['feature mean'].cuda())
            feature_mean = feature_mean.detach().data
            global_center = torch.mean(feature_mean, dim=0)

            if weight_initialization == 'fcmean':
                model.classifier.add_block[0].weight.data = feature_mean

            elif weight_initialization == 'centerspread':
                global_center = normalize(torch.mean(feature_mean, dim=0))
                new_proxies = initial_proxies(expInfo, global_center, eps=gamma)
                model.classifier.add_block[0].weight.data = new_proxies

            elif weight_initialization == 'centerprop':
                new_proxies = set_proxies_globalcenter(feature_mean, lamb=gamma)
                model.classifier.add_block[0].weight.data = new_proxies

        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            if epoch == 0:
                if loss_type == 'largemargin':
                    for param in list(set(model.parameters()).difference(criterion.parameters())):
                        param.requires_grad = False
                else:
                    for param in list(set(model.parameters()).difference(model.classifier.parameters())):
                        param.requires_grad = False
            if epoch == args.warm_epoch:
                if loss_type == 'largemargin':
                    for param in list(set(model.parameters()).difference(criterion.parameters())):
                        param.requires_grad = True
                else:
                    for param in list(set(model.parameters()).difference(model.classifier.parameters())):
                        param.requires_grad = True


            running_loss = 0.0      # loss in one iter
            running_corrects = 0    # train accuracy in one iter

            iter_ = 0               # number of iteration


            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                iter_ += 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                f, outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                # loss assigning
                loss = 0.0
                if loss_type == 'softmax':
                    loss_1 = softmax_new(expInfo, outputs, labels, mrg=1.0)
                    loss += loss_1
                elif loss_type == 'PA':
                    loss_PA = proxy_anchor(expInfo, outputs, labels)
                    loss += loss_PA
                elif loss_type == 'largemargin':
                    loss_1 = criterion(f, labels)
                    loss += loss_1

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            # inter class by hyperparameters!!

            features_after = get_feature_infos(model, expInfo, normalized_mean=False, get_more_infos=True)
            after_train_features = features_after['feature list']
            feature_move = torch.mean(torch.norm(after_train_features - before_train_features, dim=1))

            feature_mean = Variable(features_after['feature mean'].cuda())
            feature_mean = feature_mean.detach().data
            wt = model.classifier.add_block[0].weight.data

            gap = torch.mean(torch.acos(torch.diag(pairwise_similarity(feature_mean, wt))))
            inter = torch.mean(get_stretchedness(expInfo, feature_mean))

            d_intra = features_after['intra']
            d_inter = features_after['inter']
            phi = d_inter / d_intra

            if epoch % print_epoch_per == print_epoch_per-1:

                if data_type == 'Inshop':
                    results = Get_test_results_inshop(model, expInfo)

                else:
                    results, intra_test, inter_test = Get_test_results_single(model, expInfo)
                    phi_test = inter_test / intra_test

                running_corrects = running_corrects.float()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(epoch, epoch_acc, epoch_loss, results[0], results[1], results[2], results[3]))


    time_elapsed = time.time() - since

    return model


expInfo = experiment_info(data_type, image_datasets, dataloaders, model_name, class_nb, original_size, feature_size,
                          computer)

if model_name == 'RES50':
    if linear_added:
        if data_type == 'SOP':
            model = ft_net_feature_linear_added(expInfo, stride=stride, bias=True, pretrained=pretrained)
        else:
            model = ft_net_feature_linear_added(expInfo, stride=stride, pretrained=pretrained)
    else:
        model = ft_net_feature(int(class_nb))

    params_ft = []
    params_ft.append({'params': model.model.conv1.parameters(), 'lr': model_learning_rate})
    params_ft.append({'params': model.model.bn1.parameters(), 'lr': model_learning_rate})
    params_ft.append({'params': model.model.layer1.parameters(), 'lr': model_learning_rate})
    params_ft.append({'params': model.model.layer2.parameters(), 'lr': model_learning_rate})
    params_ft.append({'params': model.model.layer3.parameters(), 'lr': model_learning_rate})
    params_ft.append({'params': model.model.layer4.parameters(), 'lr': model_learning_rate})
    if expInfo.original_size != expInfo.feature_size:
        params_ft.append({'params': model.model.linear.parameters(), 'lr': layer_learning_rate})
    params_ft.append({'params': model.classifier.parameters(), 'lr': classifier_learning_rate})
    # params_ft.append({'params': model.classifier_new.parameters(), 'lr': classifier_learning_rate})

    if loss_type == 'largemargin':
        criterion = losses.SphereFaceLoss(num_classes=expInfo.class_nb, embedding_size=expInfo.feature_size, margin=4).cuda()
        params_ft.append({'params': criterion.parameters(), 'lr': classifier_learning_rate})

    if optimizer == 'SGD':
        optimizer_ft = optim.SGD(params_ft, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # elif loss_type == 'PA':
    # optimizer_ft = torch.optim.AdamW(params_ft, lr=lr4, weight_decay=1e-4)


elif model_name == 'BN':
    model = ft_net_feature_linear_added_bn(int(class_nb), feature_size=feature_size)

    # ignored_params = list(map(id, model.model.linear.parameters()))
    ignored_params = list()
    base_params = list(filter(lambda p: id(p) not in ignored_params, model.parameters()))
    if loss_type == 'largemargin':
        criterion = losses.LargeMarginSoftmaxLoss(num_classes=expInfo.class_nb, embedding_size=expInfo.feature_size, margin=4).cuda()
        base_params.append(criterion.parameters())

    if optimizer == 'SGD':
        # optimizer_ft = optim.SGD(model.parameters(), lr=fc_learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        # if loss_type == 'softmax':
        #     optimizer_ft = optim.SGD(base_params, lr=model_learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        # elif loss_type == 'PA':
        #     optimizer_ft = torch.optim.AdamW(params_ft, lr=lr4, weight_decay=1e-4)
        optimizer_ft = optim.SGD(base_params, lr=model_learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        # optimizer_ft = optim.Adam(base_params, lr=lr4, weight_decay=1e-4)



# model = load_network_path(model, '/data/ymro/AAAI2021/jhlee/CUBlabNet/net_-1.pth')

if use_gpu:
    model = model.cuda()
    # nn.DataParallel(model, device_ids=[2,3]).cuda()


if e_drop2 == 0:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=e_drop, gamma=0.1)
else:
    if loss_type == 'softmax':
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[e_drop, e_drop2], gamma=0.1)
    elif loss_type == 'PA':
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[5, 10, 15, 20, 25, 30], gamma=0.5)



model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=e_end)
del model


