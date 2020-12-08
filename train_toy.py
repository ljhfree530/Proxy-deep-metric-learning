# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib
from test_embedded import Get_test_results_single
from test_embedded import extract_feature
from test_embedded import get_id
#matplotlib.use('agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from tools import get_feature_infos
from tools import softmax_new
from tools import angle_hists
from tools import print_list_values
from tools import rotating_matrix
from tools import print_values
from tools import initial_proxies
from tools import proxy_anchor
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
from util.loss import pairwise_innerprod
import time
import os
from model import ft_net, ft_net_scratch, ft_net_feature, ft_net_feature_linear_added, ft_net_feature_linear_added_bn
from model import bn_inception_PA
from bn_inception import bn_inception
# from tensorboard_logger import configure, log_value
import json
#import visdom
import copy
import PIL


lr1 = 0.1
lr2 = 0.01
lr3 = 0.001
lr4 = 0.0001

exp_list = []
# exp_list.append({'Title': "[Ours]", 'flip': True, 'gapgmp': True, 'linear_activation': 'None', 'stride': 1, 'weight_initialization': True, 'layer_learning_rate': 0.0})
# exp_list.append({'Title': "[No Flip]", 'flip': False, 'gapgmp': True, 'linear_activation': 'None', 'stride': 1, 'weight_initialization': True, 'layer_learning_rate': 0.0})
# exp_list.append({'Title': "[Only GAP]", 'flip': True, 'gapgmp': False, 'linear_activation': 'None', 'stride': 1, 'weight_initialization': True, 'layer_learning_rate': 0.0})
# exp_list.append({'Title': "[Activate BN]", 'flip': True, 'gapgmp': True, 'linear_activation': 'BN', 'stride': 1, 'weight_initialization': True, 'layer_learning_rate': 0.0})
# exp_list.append({'Title': "[Activate ReLU]", 'flip': True, 'gapgmp': True, 'linear_activation': 'relu', 'stride': 1, 'weight_initialization': True, 'layer_learning_rate': 0.0})
# exp_list.append({'Title': "[Default Stride]", 'flip': True, 'gapgmp': True, 'linear_activation': 'None', 'stride': 0, 'weight_initialization': True, 'layer_learning_rate': 0.0})
# exp_list.append({'Title': "[Random Initial]", 'flip': True, 'gapgmp': True, 'linear_activation': 'None', 'stride': 1, 'weight_initialization': False, 'layer_learning_rate': 0.0})
# exp_list.append({'Title': "[Assign LR at linear]", 'flip': True, 'gapgmp': True, 'linear_activation': 'None', 'stride': 1, 'weight_initialization': True, 'layer_learning_rate': lr3})
exp_list.append({'Title': "[yes]", 'flip': True, 'gapgmp': True, 'linear_activation': 'None', 'stride': 1, 'weight_initialization': True, 'layer_learning_rate': lr3})


# exp_list.append({'Title': "[softmax, random, trainable]", 'flip': True, 'gapgmp': True, 'linear_activation': 'None', 'stride': 1, 'weight_initialization': False, 'layer_learning_rate': lr2})




####################################################
# Set arguments, hyperparameters
#

# base setting
computer = '87'
gpu_id = 1
data_type = 'CUB'
flip = True
base_arg_dict = {'gpu_id': gpu_id, 'data_type': data_type, 'test_flip': flip}
print(base_arg_dict)

# model setting
model_name = 'RES50'
stride = 0
gapgmp = True
linear_added = True
linear_activation = 'relu'  #'None', 'relu' or 'BN'
feature_size = 3
model_arg_dict = {'model_name': model_name, 'gapgmp': gapgmp, 'linear_added': linear_added,
                  'linear_activation': linear_activation, 'feature_size': feature_size}
if model_name == 'RES50':
    model_arg_dict = {'model_name': model_name, 'stride': stride, 'gapgmp': gapgmp,
                      'linear_added': linear_added, 'linear_activation': linear_activation, 'feature_size': feature_size}
# print(model_arg_dict)

# training setting
delta = 1e-1
pretrained = True
model_learning_rate = lr3
layer_learning_rate = lr3
classifier_learning_rate = lr3
weight_initialization = True
new_weight_initialization = False
normalized_train = True
train_arg_dict = {'model_learning_rate': model_learning_rate, 'layer_learning_rate': layer_learning_rate,
                  'classifier_learning_rate': classifier_learning_rate,
                  'Weight_initialization': weight_initialization, 'normalized_train': normalized_train}
# print(train_arg_dict)

# base loss setting
loss_type = 'PA'
softmax = True
newLossE = False
newLossM = False

rgE = 1.0
rgM = 10.0
lamb = 100.0
mg = 2.0
loss_arg_dict = {'Softmax Loss': softmax, 'New Loss': newLossM}
# print(loss_arg_dict)

# stretching setting
# type 0 : no stretch
# type 1 : stretch proxies with same degree alpha
# type 2 : stretch proxies proportionally
stretching_type = '0'
stop_stretch_epoch = 1000 # 1000 if no stop
alpha0 = 0.01
eps = 0.0
no_train_proxies = False
if stretching_type == '0':
    stop_stretch_epoch = 1000
    no_train_proxies = False
    stretching_arg_dict = {'stretching_type': stretching_type}
else:
    stretching_arg_dict = {'stretching_type': stretching_type, 'stop_stretch_epoch': stop_stretch_epoch,
                       'alpha0': alpha0, 'eps': eps, 'no_train_proxies': no_train_proxies}
# print(stretching_arg_dict)

####################################################
# Print Options
#

tops1 = []
tops = []

proxies_sim = []
intra_sim = []
inter_sim = []
prox_ft_sim = []

print_epoch_interval = 5

upper_cos = 1.0
lower_cos = 0.8
nb_interval = 20.0


####################################################
#  File/model directory
#

if data_type == 'CUB':
    data_dir = '/home/ro/FG/CUB_RT/pytorch_toy'
elif data_type == 'CAR':
    data_dir = '/home/ro/FG/STCAR_RT/pytorch'
elif data_type == 'SOP':
    data_dir = '/home/ro/FG/Stanford_Online_Products/pytorch'

# data_dir = '/data1/home/jhlee/CUB_200_2011_1/pytorch'

# dir_name = '/data/ymro/AAAI2021/base_CUB/res50_lr3_fclr2'
# dir_name = '/data/ymro/AAAI2021/jhlee/CUBlabNet/potential_loss_on'
dir_name = '/home/ro/Dropbox/jhlee'

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

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

# configure(dir_name)
# print(dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

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
if data_type == 'SOP':
    e_drop = 30
    e_drop2 = 0
    e_end = 50
else:
    e_drop = 10
    e_drop2 = 0
    e_end = 20

# Learning rates
prep_batchsize = 20
train_batchsize = 40
test_batchsize = 15

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
#
# print('Model {}'.format(model_name))
# print('train batchsize {}'.format(train_batchsize))
# print('Fc learning rate is {}'.format(fc_learning_rate))
# print('Measuring entropy with lambda {}'.format(lamb))
# if softmax:
#     print("Softmax loss")
# else:
#     print("No softmax loss")
#
# if linear_added:
#     print("Linear added with {}".format(feature_size))
# else:
#     print("No Linear added. Feature size is {}".format(feature_size))
#
# if weight_initialization:
#     print("Weight is initialized by fc_mean")
# else:
#     print("Weight is initialized randomly")
#
# if normalized_train:
#     print("normalized training")
# else:
#     print("not normalized training")
#
# if newLossE:
#     print("Entropy loss applied with rgE {}".format(rgE))
# else:
#     print("No entropy loss")
#
# if newLossM:
#     print("Stretching applied with rgM {}".format(rgM))
# else:
#     print("No stretching loss")



######################################################################
# Load Data
# ---------
#
class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im

init_resize = (256, 256)
resize = (224, 224)
if model_name == 'RES50':
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
elif model_name == 'BN':
    image_mean = [104, 117, 128]
    image_std = [1, 1, 1]


transform_prep_list = [
    RGBToBGR() if model_name == 'BN' else Identity(),
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(resize, interpolation=3),
    #transforms.RandomCrop(resize),
    transforms.ToTensor(),
    ScaleIntensities([0,1], [0,255]) if model_name == 'BN' else Identity(),
    transforms.Normalize(image_mean, image_std)
]

transform_train_list = [
    RGBToBGR() if model_name == 'BN' else Identity(),
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(resize, interpolation=3),
    #transforms.RandomCrop(resize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ScaleIntensities([0,1], [0,255]) if model_name == 'BN' else Identity(),
    transforms.Normalize(image_mean, image_std)
]

transform_val_list = [
    RGBToBGR() if model_name == 'BN' else Identity(),
    transforms.Resize(init_resize) if data_type == 'CUB' else transforms.Resize(resize),  # Image.BICUBIC
    transforms.CenterCrop(resize) if data_type == 'CUB' else Identity(),
    transforms.ToTensor(),
    ScaleIntensities([0,1], [0,255]) if model_name == 'BN' else Identity(),
    transforms.Normalize(image_mean, image_std),
]

#
# init_resize = (256, 256)
# resize = (224, 224)
#
# transform_prep_list = [
#     # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
#     transforms.Resize(resize, interpolation=3),
#     #transforms.RandomCrop(resize),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]
#
# transform_train_list = [
#     # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
#     transforms.Resize(resize, interpolation=3),
#     #transforms.RandomCrop(resize),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]
#
# transform_val_list = [
#     transforms.Resize(init_resize, interpolation=3),  # Image.BICUBIC
#     transforms.CenterCrop(resize),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]






# print(transform_train_list)
data_transforms = {
    'prep' : transforms.Compose(transform_prep_list),
    'train': transforms.Compose(transform_train_list),
    'test': transforms.Compose(transform_val_list),
}



image_datasets = {}
image_datasets['prep'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['prep'])
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                               data_transforms['train'])
image_datasets['test'] = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                               data_transforms['test'])

dataloaders = {}
dataloaders['prep'] = torch.utils.data.DataLoader(image_datasets['prep'], batch_size=prep_batchsize, shuffle=False, num_workers=16)
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=train_batchsize, shuffle=True, num_workers=16)

dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size= test_batchsize, shuffle=False, num_workers=8)



dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes
class_nb = len(class_names)

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))


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

expInfo = experiment_info(data_type, image_datasets, dataloaders, model_name, class_nb, original_size, feature_size, computer)



def train_model(model, optimizer, scheduler, num_epochs=25):
    since = time.time()

    midprocess = {'intra': [], 'inter': [], 'stretched': [], 'attached': []}
    process_print = False

    # normalizing proxies
    wt = model.classifier.add_block[0].weight.data
    model.classifier.add_block[0].weight.data = normalize(wt)


    # Initializing weight as fc_mean, if weight_initialization is True.

    # weight = model.classifier.add_block[0].weight.data
    # _, initial_feature_mean, _ = get_feature_info(model, feature_size, normalized_train)
    # weight_sim, _ = potential_normalized_softmax(weight)
    # fc_sim, _ = potential_normalized_softmax(initial_feature_mean)
    # sim_hists(weight_sim)
    # sim_hists(fc_sim)

    # feature_list, feature_mean, labels = get_feature_infos(model, feature_size, normalized_train)
    # feature_list = feature_list.cuda()

    # results = Get_test_results_single(image_datasets['test'], dataloaders['test'], model, f_size=feature_size,
    #                                   datatype=data_type, flip=flip)
    # results1 = Get_test_results_single(image_datasets['test'], dataloaders['test'], model, f_size=original_size,
    #                                    is_before_linear=True, datatype=data_type, flip=flip)
    # print('512 : {:.4f} {:.4f} {:.4f} {:.4f}'.format(results[0], results[1], results[2], results[3]))
    # print('2048: {:.4f} {:.4f} {:.4f} {:.4f}'.format(results1[0], results1[1], results1[2], results1[3]))

    wt_prev = None
    wt_curr = None
    ft_prev = None
    ft_curr = None


    # total_feature, initial_feature_mean, _ = get_feature_info(model, feature_size, normalized_train)
    # initial_feature_mean = Variable(initial_feature_mean.cuda())
    # model.classifier.add_block[0].weight.data = initial_feature_mean
    wt_curr = model.classifier.add_block[0].weight.data
    features_after = get_feature_infos(model, expInfo, normalized_mean=False, get_more_infos=True)
    # print_infos(features_after)
    feature_list_not_normed = features_after['feature list not normed'].cuda()
    feature_mean = Variable(features_after['feature mean'].cuda())
    feature_mean = feature_mean.detach().data
    if weight_initialization:
        new_proxies = initial_proxies(expInfo, normalize(torch.mean(feature_mean, dim=0)), eps=delta)
        model.classifier.add_block[0].weight.data = new_proxies
    ft_curr = features_after['feature list']



    features_after_initialized = get_feature_infos(model, expInfo, normalized_mean=False, get_more_infos=True, info_dict = features_after)
    # print_infos(features_after_initialized)

    feature_center = normalize(torch.mean(feature_list_not_normed, dim=0))
    origin = torch.zeros(feature_size)
    origin[feature_size-1] = 1.0
    rot = rotating_matrix(feature_center, origin)

    wt_curr = model.classifier.add_block[0].weight.data

    rotted_wt = torch.mm(wt_curr, rot.t())
    for ind in range(class_nb):
        mask = (features_after_initialized['labels'] == ind)
        if expInfo.computer == '58':
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        features_single_class = features_after_initialized['feature list'][mask]
        rotted_feature = torch.mm(features_single_class, rot.t())
        print()
        for vec in rotted_feature:
            print_values(vec, name='')
        print()
        print_values(rotted_wt[ind], name='')
        # features_single_class_ = features_single_class - feature_center
        # feature_mean_ = feature_mean[0] - feature_center
        # r_zero = torch.acos(torch.sum(feature_mean[0] * feature_center))
        # r_mean = torch.acos(torch.sum(feature_mean[ind] * feature_center))
        #
        # great_circle_mean = feature_mean[ind] * (1.0 / torch.sin(r_mean)) - feature_center * (1.0 / torch.tan(r_mean))
        # great_circle_mean_zero = feature_mean[0] * (1.0 / torch.sin(r_zero)) - feature_center * (1.0 / torch.tan(r_zero))
        #
        # t_mean = torch.acos(torch.sum(great_circle_mean * great_circle_mean_zero))
        #
        # r = torch.acos(pairwise_similarity(feature_center, features_single_class).squeeze())
        # great_circle_single = (features_single_class * (1.0 / torch.sin(r))[:, None] - feature_center[None, :] * (1.0 / torch.tan(r))[:, None])
        # t = torch.acos(pairwise_similarity(great_circle_single, great_circle_mean_zero)).squeeze()

        # print("{:.3f} {:.3f}".format(r_mean, t_mean))
        # print("{:.3f} {:.3f}".format(r_mean, t_mean))
        # print_values(r, name='')
        # print_values(t, name='')

    standard_vector = feature_center




    if new_weight_initialization:
        features_before = get_feature_infos(model, expInfo, is_before_linear=True, normalized=normalized_train)
        feature_list = features_before['feature list'].cuda()
        feature_mean = Variable(features_before['feature mean'].cuda())

        w = normalize(model.classifier.add_block[0].weight.data)[features_before['labels']]
        fft_inv = torch.inverse(torch.mm(feature_list.t(), feature_list))
        fwt = torch.mm(feature_list.t(), w)
        model.model.linear.weight.data = torch.mm(fft_inv, fwt).t()

    results = Get_test_results_single(model, expInfo, flip=flip)
    results1 = Get_test_results_single(model, expInfo, flip=flip, is_before_linear=True)

    # print('512 : {:.4f} {:.4f} {:.4f} {:.4f}'.format(results[0], results[1], results[2], results[3]))
    # print('2048: {:.4f} {:.4f} {:.4f} {:.4f}'.format(results1[0], results1[1], results1[2], results1[3]))
    # print('------')

    # get_individual_sims(feature_list, labels, model.classifier.add_block[0].weight.data)

    for epoch in range(num_epochs):
        # print('-' * 10)
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # _, pre_feature_mean, _ = get_feature_info(model, feature_size, normalized_train)

        # wt = model.classifier.add_block[0].weight.data.cuda()
        # model.classifier.add_block[0].weight.data = stretching_proxies(wt, alpha0, eps, stretching_type).cuda()

        if epoch == 0:
            for param in list(set(model.parameters()).difference(model.classifier.parameters())):
                param.requires_grad = False
        if epoch == 5:
            for param in list(set(model.parameters()).difference(model.classifier.parameters())):
                param.requires_grad = True


        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_1 = 0.0
            running_loss_E = 0.0
            running_loss_M = 0.0
            running_corrects = 0

            iter_ = 0

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

                # Softmax loss

                # loss_1 = criterion(outputs, labels)
                # loss_1 = softmax_new(expInfo, labels, mrg=1.0)
                # criterion_test = Softmax_new(nb_classes=class_nb, sz_embed=feature_size, mrg=mg)
                # loss_test = criterion_test(outputs, labels)

                # Pfloss
                # wt = model.classifier.add_block[0].weight.detach().data
                wt = model.classifier.add_block[0].weight.data
                wt_ = model.classifier.add_block[0].weight.data
                wt_mean_ = torch.mean(wt_, dim=0)
                # Ploss = potential_normalized_exp(wt, alpha)
                # sims_vec, entropy = potential_normalized_softmax(wt, l=lamb)
                # Eloss = np.log( class_nb * (class_nb - 1)/2.0 ) - entropy


                Mloss = torch.mean(pairwise_similarity(f, wt_mean_))
                loss_PA = proxy_anchor(expInfo, outputs, labels)

                loss_1 = softmax_new(expInfo, outputs, labels, mrg=1.0)

                # loss_1 = 1.0 - pairwise_similarity(f, wt[labels], getMean=True)


                # wt = wt.cpu()
                # Eloss = torch.std(sims_vec)


                # loss_1 = 1.0 - pairwise_similarity(f, wt[labels], getMean=True)

                # Mloss = torch.mean(sims_vec)
                # Mloss = smooth_max(sims_vec)


                # loss_new = torch.pow(ft - wt, 2.0).sum()
                loss = 0.0
                if loss_type == 'softmax':
                    loss_1 = softmax_new(expInfo, outputs, labels, mrg=1.0)
                    loss += loss_1
                elif loss_type == 'PA':
                    loss_PA = proxy_anchor(expInfo, outputs, labels)
                    loss += loss_PA


                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if no_train_proxies:
                    model.classifier.add_block[0].weight.data = wt
                    model.classifier.add_block[0].weight.data = model.classifier.add_block[0].weight.data.cuda()


                running_loss += loss.data
                running_loss_1 += loss_1.data
                # running_loss_E += Eloss.data
                running_loss_M += Mloss.data
                running_corrects += torch.sum(preds == labels.data)



            if epoch == 0:
                results, test_center = Get_test_results_single(model, expInfo, flip=flip, print_features=True)
            else:
                results = Get_test_results_single(model, expInfo, print_features=True, feature_center=test_center)
            # results1 = Get_test_results_single(image_datasets['test'], dataloaders['test'], model, f_size=original_size, is_before_linear=True, datatype=data_type, flip=flip, model_final=original_size)
            # print(results)
            running_corrects = running_corrects.float()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_1 = running_loss_1 / dataset_sizes[phase]
            epoch_loss_E = running_loss_E / iter_
            epoch_loss_M = running_loss_M / iter_
            epoch_acc = running_corrects / dataset_sizes[phase]

            # feature_list, feature_mean, _ = get_feature_info(model, feature_size, normalized_train)
            # feature_mean_norms = torch.norm(feature_mean, dim=1)

            ####
            # feature_list, feature_mean, labels = get_feature_infos(model, feature_size, normalized_train)
            # feature_list = feature_list.cuda()
            # weight = normalize(model.classifier.add_block[0].weight.data.cpu())
            # epoch_sims_vec, epoch_entropy = potential_normalized_softmax(weight, l=lamb)

            ####


            # feature_mean = Variable(feature_mean.cuda())
            # weight_norms = torch.norm(weight, dim=1)
            # weight_mean_norm = torch.norm(torch.mean(weight, dim=1))

            # feature_weight_sim = torch.diag(pairwise_similarity(feature_mean, weight))

            # feature_newweight_sim = torch.diag(pairwise_similarity(feature_mean, weight - torch.mean(weight, dim=0)))

            # feature_mean_sims_vec, _ = potential_normalized_softmax(feature_mean)


            ########
            # print('Epoch {}/{} Sloss {:.4f} Eloss {:.4f} Mloss {:.4f} Acc {:.4f} Entropy {:.4f} Stdev {:.4f}'.
            #       format(epoch, num_epochs - 1, epoch_loss_1, epoch_loss_E, epoch_loss_M, epoch_acc, epoch_entropy, torch.std(epoch_sims_vec)))
            # print('Test with features before linear: top-1 {:.4f} top-2 {:.4f} top-4 {:.4f} top-8 {:.4f}'.format(results1[0], results1[1], results1[2], results1[3]))
            # print('Test with features after  linear: top-1 {:.4f} top-2 {:.4f} top-4 {:.4f} top-8 {:.4f}'.format(results[0], results[1], results[2], results[3]))
            # sim_hists(epoch_sims_vec, upper=1.0, lower=0.7, N=20.0, name="proxies: ")
            # get_individual_sims(feature_list, labels, model.classifier.add_block[0].weight.data)
            ######

            features_after = get_feature_infos(model, expInfo, normalized_mean=False, get_more_infos=True)
            feature_mean = features_after['feature mean']

            wt_prev = wt_curr
            wt_curr = model.classifier.add_block[0].weight.data
            # wt_del = torch.mean(torch.acos(torch.diagonal(pairwise_similarity(wt_prev, wt_curr))))

            ft_prev = ft_curr
            ft_curr = features_after['feature list']
            ft_del = torch.mean(torch.acos(torch.diagonal(pairwise_similarity(ft_prev, ft_curr))))

            print('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(epoch, epoch_acc, 0, ft_del, results[0]))
            # print('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(epoch, results[0], results[1], results[2], results[3]))
            # if epoch == e_end - 1:
            #     print('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(epoch, results1[0], results1[1], results1[2], results1[3]))

            # log_value('train_loss', epoch_loss, epoch)
            # log_value('train_acc', epoch_acc, epoch)

            midprocess['intra'].append(angle_hists(features_after['intra']))
            midprocess['inter'].append(angle_hists(features_after['inter'], upper=20.0))
            midprocess['stretched'].append(angle_hists(features_after['stretched']))
            midprocess['attached'].append(angle_hists(features_after['attached']))

            # feature_list_not_normed = features_after['feature list not normed']
            # feature_mean = features_after['feature mean'].cuda()
            # feature_center = standard_vector
            # proxies = model.classifier.add_block[0].weight.data
            #

            rotted_wt = torch.mm(wt_curr, rot.t())
            print('train features')
            for ind in range(class_nb):
                mask = (features_after['labels'] == ind)
                if expInfo.computer == '58':
                    mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
                features_single_class = features_after['feature list'][mask]
                rotted_feature = torch.mm(features_single_class, rot.t())
                print(ind)
                for vec in rotted_feature:
                    print_values(vec, name='')
                print("epoch {}".format(epoch))
                print_values(rotted_wt[ind], name='')
            #     # features_single_class_ = features_single_class - feature_center
            #     # feature_mean_ = feature_mean[0] - feature_center
            #     r_zero = torch.acos(torch.sum(feature_mean[0] * feature_center))
            #     r_mean = torch.acos(torch.sum(feature_mean[ind] * feature_center))
            #     r_prox = torch.acos(torch.sum(proxies[ind] * feature_center))
            #     great_circle_mean = feature_mean[ind] * (1.0 / torch.sin(r_mean)) - feature_center * (
            #                 1.0 / torch.tan(r_mean))
            #     great_circle_prox = proxies[ind] * (1.0 / torch.sin(r_prox)) - feature_center * (
            #             1.0 / torch.tan(r_prox))
            #     great_circle_mean_zero = feature_mean[0] * (1.0 / torch.sin(r_zero)) - feature_center * (
            #                 1.0 / torch.tan(r_zero))
            #     t_mean = torch.acos(torch.sum(great_circle_mean * great_circle_mean_zero))
            #     t_prox = torch.acos(torch.sum(great_circle_prox * great_circle_mean_zero))
            #     r = torch.acos(pairwise_similarity(feature_center, features_single_class).squeeze())
            #     great_circle_single = (features_single_class * (1.0 / torch.sin(r))[:, None]
            #                            - feature_center[None, :] * (1.0 / torch.tan(r))[:, None])
            #     t = torch.acos(pairwise_similarity(great_circle_single, great_circle_mean_zero)).squeeze()
            #
            #     print("{:.3f} {:.3f}".format(r_mean, t_mean))
            #     print("{:.3f} {:.3f}".format(r_prox, t_prox))
            #     print_values(r, name='')
            #     print_values(t, name='')



            # sim_hists(feature_mean_sims_vec, name="fc mean: ")

            # last_model_wts = model.state_dict()
            # save_network(model, epoch)
            if process_print:
                print_list_values(midprocess['intra'], name='intra')
                print_list_values(midprocess['stretched'], name='stretched')
                print_list_values(midprocess['attached'], name='attached')
                print_list_values(midprocess['inter'], name='inter')

    # print_list_values(midprocess['intra'], name='intra')
    # print_list_values(midprocess['stretched'], name='stretched')
    # print_list_values(midprocess['attached'], name='attached')
    # print_list_values(midprocess['inter'], name='inter')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model




##################
#
#  Model defining

def load_network_path(network, save_path):
    state_dict_ = torch.load(save_path)
    network.load_state_dict(state_dict_['model_state_dict'])
    print("Loaded")
    return network



best_path = '/home/ro/cub_bn_inception_best.pth'
nb_exp = 5

# for i in range(len(exp_list)):
while True:
    exp = exp_list[0]
    print(exp)

    flip = exp['flip']
    gapgmp = exp['gapgmp']
    linear_activation = exp['linear_activation']
    stride = exp['stride']
    weight_initialization = exp['weight_initialization']
    layer_learning_rate = exp['layer_learning_rate']

    tops1 = []
    tops = []

    proxies_sim = []
    intra_sim = []
    inter_sim = []
    prox_ft_sim = []

    if model_name == 'RES50':
        if linear_added:
            if data_type == 'SOP':
                model = ft_net_feature_linear_added(expInfo, stride=stride, bias=True, pretrained=pretrained)
            else:
                model = ft_net_feature_linear_added(expInfo, stride=stride, pretrained=pretrained)
        else:
            model = ft_net_feature(int(class_nb), normalized=normalized_train)

        params_ft = []
        params_ft.append({'params': model.model.conv1.parameters(), 'lr': model_learning_rate})
        params_ft.append({'params': model.model.bn1.parameters(), 'lr': model_learning_rate})
        params_ft.append({'params': model.model.layer1.parameters(), 'lr': model_learning_rate})
        params_ft.append({'params': model.model.layer2.parameters(), 'lr': model_learning_rate})
        params_ft.append({'params': model.model.layer3.parameters(), 'lr': model_learning_rate})
        params_ft.append({'params': model.model.layer4.parameters(), 'lr': model_learning_rate})
        if expInfo.original_size != expInfo.feature_size:
            params_ft.append({'params': model.model.linear.parameters(), 'lr': layer_learning_rate})
            if linear_activation == 'BN':
                params_ft.append({'params': model.model.bn_final.parameters(), 'lr': model_learning_rate})
        params_ft.append({'params': model.classifier.parameters(), 'lr': classifier_learning_rate})
        # params_ft.append({'params': model.classifier_new.parameters(), 'lr': classifier_learning_rate})

        if loss_type == 'softmax':
            optimizer_ft = optim.SGD(params_ft, momentum=0.9, weight_decay=5e-4, nesterov=True)
        elif loss_type == 'PA':
            optimizer_ft = torch.optim.AdamW(params_ft, lr=lr4, weight_decay=1e-4)


    elif model_name == 'BN':
        model = ft_net_feature_linear_added_bn(int(class_nb), feature_size=feature_size)

        # ignored_params = list(map(id, model.model.linear.parameters()))
        ignored_params = list()
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

        # optimizer_ft = optim.SGD(model.parameters(), lr=fc_learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        if loss_type == 'softmax':
            optimizer_ft = optim.SGD(base_params, lr=model_learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        elif loss_type == 'PA':
            optimizer_ft = torch.optim.AdamW(params_ft, lr=lr4, weight_decay=1e-4)

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

    print()
    if model == 0:
        continue
    else:
        break


