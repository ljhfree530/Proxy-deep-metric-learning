# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib
from test_embedded import Get_test_results_single
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
from PIL import Image
from util.loss import potential
from util.loss import potential_normalized
from util.loss import CompactnessLoss
from util.loss import EntropyLoss
from util.loss import HLoss
from util.loss import HLoss_normalized
import time
import os
from model import ft_net, ft_net_scratch, ft_net_feature
from tensorboard_logger import configure, log_value
import json
#import visdom
import copy


######################################################################
# Options

# data_dir = '/home/ro/FG/STCAR_RT/pytorch'
data_dir = '/home/ro/FG/CUB_RT/pytorch'

# data_dir = '/data1/home/jhlee/CUB_200_2011_1/pytorch'
# dddd

# dir_name = '/data/ymro/AAAI2021/base_CUB/res50_lr3_fclr2'
dir_name = '/data/ymro/AAAI2021/jhlee/CUBlabNet'

e_drop = 10
e_drop2 = 0
e_end = 20

prep_batchsize = 30
train_batchsize = 40
test_batchsize = 10
configure(dir_name)
print(dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


use_gpu = True
# os.environ['CUDA_VISIBLE_DEVICES']='1'
gpu_id = 0
gpu_ids = []
gpu_ids.append(gpu_id)
# set gpu id2
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
print(gpu_ids[0])

rg = 0.0
lr1 = 0.1
lr2 = 0.01
lr3 = 0.001
lr4 = 0.0001

######################################################################
# Load Data
# ---------
#
#init_resize = (256, 256)
resize = (224, 224)

transform_prep_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(resize, interpolation=3),
    #transforms.RandomCrop(resize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
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
#     transforms.Resize(resize, interpolation=3),  # Image.BICUBIC
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ]


print(transform_prep_list)
data_transforms = {
    'prep' : transforms.Compose(transform_prep_list),
    # 'train': transforms.Compose(transform_train_list),
    # 'test': transforms.Compose(transform_val_list),
}



image_datasets = {}
image_datasets['prep'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['prep'])
dataloaders = {}
dataloaders['prep'] = torch.utils.data.DataLoader(image_datasets['prep'], batch_size=prep_batchsize, shuffle=False, num_workers=16)


dataset_sizes = {x: len(image_datasets[x]) for x in ['prep']}
class_names = image_datasets['prep'].classes

use_gpu = torch.cuda.is_available()

# inputs, classes = next(iter(dataloaders['train']))




######################################################################
def prep_model(model):
    feature_list = [None] * len(class_names)
    meanFeature = torch.zeros((len(class_names), 2048))
    numDataByClass = torch.ones((len(class_names)))
    model.eval()
    index = 0

    for data in dataloaders['prep']:
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        f, _ = model(inputs)
        f = f.data.cpu()

        mask = (labels == index)
        if torch.sum(mask) == 0:
            index += 1
            feature_list[index] = f

        else:
            if type(feature_list[index]) == type(None):
                feature_list[index] = f[mask]
            else:
                torch.cat((feature_list[index], f[mask]))

            if torch.sum(mask) < len(labels):
                index += 1
                new_mask = (labels == index)
                feature_list[index] = f[new_mask]

    for index in range(len(class_names)):
        meanFeature[index] = torch.mean(feature_list[index], dim=0)

    print(potential_normalized(meanFeature))
    model.classifier.add_block[1].weight.data = meanFeature

    return model



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
    print(save_path)

        # nn.DataParallel(network, device_ids=[2,3]).cuda()


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

model = ft_net_feature(int(len(class_names)))

if use_gpu:
    model = model.cuda()
    # nn.DataParallel(model, device_ids=[2,3]).cuda()

criterion = nn.CrossEntropyLoss()
criterion_potential = HLoss_normalized(reg=rg)
# criterion_entropy = EntropyLoss(margin=0.5,pc_direct=True)

params_ft = []
params_ft.append({'params': model.model.conv1.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.bn1.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer1.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer2.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer3.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer4.parameters(), 'lr': lr3})
params_ft.append({'params': model.classifier.parameters(), 'lr': lr2})

optimizer_ft = optim.SGD(params_ft, momentum=0.9, weight_decay=5e-4, nesterov=True)


if e_drop2 == 0:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=e_drop, gamma=0.1)
else:
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[e_drop,e_drop2], gamma=0.1)

model = prep_model(model)
save_network(model, -1)
# model = train_model(model, criterion, criterion_potential, optimizer_ft, exp_lr_scheduler,
#                     num_epochs=e_end)
