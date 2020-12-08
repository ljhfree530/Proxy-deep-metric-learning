# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import os
import PIL
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



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



def image_loader(model_name, data_type, data_dir, prep_batchsize, train_batchsize, test_batchsize):
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
        transforms.CenterCrop(resize) if (data_type in ['CUB', 'CAR']) else Identity(),
        transforms.ToTensor(),
        ScaleIntensities([0,1], [0,255]) if model_name == 'BN' else Identity(),
        transforms.Normalize(image_mean, image_std),
    ]




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
    if data_type == 'Inshop':
        image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                                       data_transforms['test'])
        image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                                       data_transforms['test'])
    else:
        image_datasets['test'] = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                                       data_transforms['test'])



    dataloaders = {}
    dataloaders['prep'] = torch.utils.data.DataLoader(image_datasets['prep'], batch_size=prep_batchsize, shuffle=False, num_workers=16)
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=train_batchsize, shuffle=True, num_workers=16)

    if data_type == 'Inshop':
        dataloaders['query'] = torch.utils.data.DataLoader(image_datasets['query'], batch_size=test_batchsize, shuffle=False, num_workers=8)
        dataloaders['gallery'] = torch.utils.data.DataLoader(image_datasets['gallery'], batch_size= test_batchsize, shuffle=False, num_workers=8)
    else:
        dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size= test_batchsize, shuffle=False, num_workers=8)

    return image_datasets, dataloaders
