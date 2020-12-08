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
import random
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib
from test_embedded import Get_test_results_single
from test_embedded import extract_feature
from test_embedded import get_id
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
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
import sklearn.preprocessing



def print_values(vector, name='Name: '):
    epoLfmt = ['{:.3f}'] * len(vector)
    epoLfmt = ' '.join(epoLfmt)
    values = []
    for i in range(len(vector)):
        values.append(vector[i])
    epoLfmt = name + epoLfmt
    print(epoLfmt.format(*values))

def print_list_values(vector_list, name='Name'):
    print(name)
    for vector in vector_list:
        epoLfmt = ['{:.3f}'] * len(vector)
        epoLfmt = ' '.join(epoLfmt)
        values = []
        for i in range(len(vector)):
            values.append(vector[i])
        epoLfmt = epoLfmt
        print(epoLfmt.format(*values))

def sim_hists(sim_vec, upper = 1.0, lower = -1.0, N = 50.0, name='Name: '):
    hist = []
    length = upper - lower
    for i in range(int(N)):
        bool_ = (sim_vec <= (upper - length * i / N)) & (sim_vec > (upper - length * (i+1) / N))
        hist.append(torch.sum(bool_))
    bool_ = (sim_vec <= lower)
    hist.append(torch.sum(bool_))
    print_values(hist, name=name)

def angle_hists(angles, upper = 2.0, lower = 0.0, N=20.0):
    hist = []
    length = upper - lower
    for i in range(int(N)):
        bool_ = (angles > (lower + length * i / N)) & (angles <= (lower + length * (i+1) / N))
        hist.append(torch.sum(bool_))
    bool_ = (angles > upper)
    hist.append(torch.sum(bool_))
    return hist

def angle_list_hists(angles_list, upper = 2.0, lower = 0.0, N=20.0):
    res_list = []
    for angles in angles_list:
        hist = []
        length = upper - lower
        for i in range(int(N)):
            bool_ = (angles > (lower + length * i / N)) & (angles <= (lower + length * (i+1) / N))
            hist.append(torch.sum(bool_))
        bool_ = (angles > upper)
        hist.append(torch.sum(bool_))
        res_list.append(hist)
    return res_list

# def get_feature_info(model, expInfo, normalized=False):
#     '''
#     :param model:
#     :return feature_list, feature_mean, feature_center:
#     '''
#     feature_list = [None] * expInfo.class_nb
#     feature_mean= torch.zeros((expInfo.class_nb, expInfo.feature_size))
#     feature_num = torch.zeros(expInfo.class_nb)
#     feature_center = torch.zeros(expInfo.feature_size)
#     model.eval()
#     index = 0
#
#     for data in expInfo.dataloaders['prep']:
#         inputs, labels = data
#         inputs = Variable(inputs.cuda())
#         labels = Variable(labels.cuda())
#         f, _ = model(inputs)
#         f = f.data.cpu()
#
#         mask = (labels == index)
#         # print(labels)
#         # print(mask)
#         if torch.sum(mask) == 0:
#             index += 1
#             feature_list[index] = f
#
#         else:
#             if type(feature_list[index]) == type(None):
#                 feature_list[index] = f[mask]
#             else:
#                 feature_list[index] = torch.cat((feature_list[index], f[mask]))
#
#             if torch.sum(mask) < len(labels):
#                 index += 1
#                 new_mask = (labels == index)
#                 feature_list[index] = f[new_mask]
#
#
#
#     for index in range(expInfo.class_nb):
#         if normalized:
#             feature_list[index] = normalize(feature_list[index])
#         feature_mean[index] = torch.mean(feature_list[index], dim=0)
#         feature_num[index] = len(feature_list[index])
#
#     if normalized:
#         feature_mean = normalize(feature_mean)
#
#     for index in range(expInfo.class_nb):
#         feature_center += feature_mean[index] * feature_num[index]
#     feature_center /= torch.sum(feature_num)
#
#     return feature_list, feature_mean, feature_center


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def softmax_new(expInfo, outputs, labels, mrg):
    P_one_hot = binarize(T=labels, nb_classes=expInfo.class_nb)
    corr_mat = torch.ones_like(P_one_hot) + P_one_hot * (mrg - 1)
    new_outputs = outputs * corr_mat
    loss = - torch.sum(P_one_hot * new_outputs) + torch.sum(torch.log(torch.sum(torch.exp(new_outputs), dim=1)))

    # test = torch.sum(torch.exp(P_one_hot * new_outputs), dim=1) / torch.sum(torch.exp(new_outputs), dim=1)
    prep = torch.exp(new_outputs) # torch.exp(new_outputs) is original.
    test = torch.sum(P_one_hot * prep, dim=1) / torch.sum(prep, dim=1)
    loss = - torch.sum(torch.log(test))

    return loss / len(labels)

def triplet_new(expInfo, outputs, labels, mrg):
    P_one_hot = binarize(T=labels, nb_classes=expInfo.class_nb)
    corr_mat = torch.ones_like(P_one_hot) + P_one_hot * (mrg - 1)
    new_outputs = outputs * corr_mat
    loss = - torch.sum(P_one_hot * new_outputs) + torch.sum(torch.log(torch.sum(torch.exp(new_outputs), dim=1)))

    # test = torch.sum(torch.exp(P_one_hot * new_outputs), dim=1) / torch.sum(torch.exp(new_outputs), dim=1)
    prep = torch.exp(new_outputs) # torch.exp(new_outputs) is original.
    test = torch.sum(P_one_hot * prep, dim=1) / torch.sum(prep, dim=1)
    loss = - torch.sum(torch.log(test))

    return loss / len(labels)


def proxy_anchor(expInfo, outputs, labels, mrg=0.1, alpha=32):
    P_one_hot = binarize(T=labels, nb_classes=expInfo.class_nb)
    N_one_hot = 1 - P_one_hot

    pos_exp = torch.exp(-alpha * (outputs - mrg))
    neg_exp = torch.exp(alpha * (outputs + mrg))

    with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
        dim=1)  # The set of positive proxies of data in the batch
    num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies


    P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
    N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

    pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
    neg_term = torch.log(1 + N_sim_sum).sum() / expInfo.class_nb
    loss = pos_term + neg_term

    return loss



def stretch_weighted(expInfo, proxies, weight):
    angles = get_stretchedness(expInfo, proxies)

    return torch.mean(weight * angles)

def initial_proxies(expInfo, mean_vector, eps=1e-2):
    mean_vectors = mean_vector.repeat(expInfo.class_nb).reshape(expInfo.class_nb, expInfo.feature_size)

    return normalize(mean_vectors + (eps * torch.rand(mean_vectors.size())).cuda())

# class Softmax_new(torch.nn.Module):
#     def __init__(self, nb_classes, sz_embed, mrg=1.0, alpha=32):
#         torch.nn.Module.__init__(self)
#         # Proxy Anchor Initialization
#         # self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
#         # nn.init.kaiming_normal_(self.proxies, mode='fan_out')
#
#         self.nb_classes = nb_classes
#         self.sz_embed = sz_embed
#         self.mrg = mrg
#         self.alpha = alpha
#
#     def forward(self, outputs, T):
#         P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
#
#         loss = - self.mrg * torch.sum(P_one_hot * outputs) + torch.sum(torch.log(torch.sum(torch.exp(outputs), dim=1)))
#
#         return loss / len(T)
def get_feature_infos(model, expInfo, is_before_linear=False, normalized_mean=True, get_more_infos=False, info_dict=None):
    res_dict = {}

    proxies = model.classifier.add_block[0].weight.data

    query_path = expInfo.image_datasets['prep'].imgs
    query_label = np.array(get_id(query_path))
    if info_dict is None:
        query_feature_not_normed = extract_feature(model, expInfo, dataloaders=expInfo.dataloaders['prep'], is_before_linear=is_before_linear, flip=False).cuda()
        query_feature = normalize(query_feature_not_normed)
    else:
        res_dict = info_dict
        query_feature_not_normed = info_dict['feature list not normed']
        query_feature = info_dict['feature list']

    if is_before_linear:
        query_feature_mean = torch.zeros((expInfo.class_nb, expInfo.original_size))
        query_feature_mean_not_normed = torch.zeros((expInfo.class_nb, expInfo.original_size))
    else:
        query_feature_mean = torch.zeros((expInfo.class_nb, expInfo.feature_size))
        query_feature_mean_not_normed = torch.zeros((expInfo.class_nb, expInfo.feature_size))

    for index in range(expInfo.class_nb):
        mask = (query_label == index)
        if expInfo.computer == '58':
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        if normalized_mean:
            mean = torch.mean(query_feature[mask], dim=0)
        else:
            mean = torch.mean(query_feature_not_normed[mask], dim=0)
        query_feature_mean[index] = normalize(mean)
        query_feature_mean_not_normed[index] = mean

    global_center = torch.mean(query_feature_mean, dim=0)
    if normalized_mean:
        global_center = normalize(global_center)

    res_dict['feature list not normed'] = query_feature_not_normed
    res_dict['feature list'] = query_feature
    res_dict['feature mean'] = query_feature_mean
    res_dict['global center'] = global_center
    res_dict['labels'] = query_label

    if get_more_infos:
        intra_angles, inter = get_intra_inter(expInfo, query_feature, query_feature_mean_not_normed, query_label)
        stretched_angles = get_stretchedness(expInfo, proxies)
        attached_angles = get_attachedness(expInfo, query_feature, query_label, proxies)

        res_dict['intra'] = intra_angles
        res_dict['inter'] = inter
        res_dict['stretched'] = stretched_angles
        res_dict['attached'] = attached_angles

    return res_dict

def print_infos(info_dict):
    print_values(angle_hists(info_dict['intra']), name="Intra     :")
    # print_values(info_dict['intra'], name="Intra     :")
    print_values(angle_hists(info_dict['stretched']), name="Stretched :")
    # print_values(info_dict['stretched'], name="Stretched :")
    print_values(angle_hists(info_dict['attached']), name="Attached  :")
    # print_values(info_dict['attached'], name="Attached  :")
    print_values(angle_hists(info_dict['inter']), name="Inter     :")
    # print_values(info_dict['inter'], name="Inter     :")

def get_intra_inter(expInfo, query_feature, query_feature_mean, query_label):
    # intra_angles = []
    # for index in range(expInfo.class_nb):
    #     mask = (query_label == index)
    #     if expInfo.computer == '58':
    #         mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).cuda()
    #     sims = pairwise_similarity(pairwise_similarity(query_feature[mask]))
    #     length = query_feature[mask].size(0)
    #     sims_without_diag = sims[torch.triu(torch.ones(length, length), 1) == 1]
    #     angle = torch.acos(sims_without_diag)
    #     # angle = torch.acos(pairwise_similarity(query_feature[mask]) - torch.eye(query_feature[mask].size(0)).cuda())
    #     intra_angles.append(torch.mean(angle))
    #
    # intra_angles = torch.FloatTensor(intra_angles).cuda()
    # mean_angles = torch.acos(pairwise_similarity(query_feature_mean, query_feature_mean)).cuda()
    # # intra_sum = torch.unsqueeze(intra_angles, 0) + torch.unsqueeze(intra_angles, 1)
    # # inter = 2.0 * intra_sum / mean_angles
    # # inter_ = inter[torch.triu(torch.ones(class_nb, class_nb), 1) == 1]
    #
    # inter_ = mean_angles[torch.triu(torch.ones(expInfo.class_nb, expInfo.class_nb), 1) == 1]

    query_feature = query_feature.cuda()
    query_feature_mean = query_feature_mean.cuda()
    # l2-norm version
    intra_dists = []
    for index in range(expInfo.class_nb):
        mask = (query_label == index)
        if expInfo.computer == '58':
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).cuda()
        norms = torch.norm(query_feature[mask] - query_feature_mean[index], dim=1)
        avgnorm = torch.mean(norms)
        intra_dists.append(avgnorm)

    inter_sum = torch.tensor(0.0).cuda()
    for index in range(expInfo.class_nb):
        norms = torch.norm(query_feature_mean - query_feature_mean[index], dim=1)
        inter_sum += torch.sum(norms)
    inter_avg = inter_sum / (expInfo.class_nb * (expInfo.class_nb - 1))

    return torch.mean(avgnorm), inter_avg

def get_stretchedness(expInfo, proxies):
    # angles = []
    # for index in range(class_nb):
    #     angle = torch.acos(pairwise_similarity(proxies[index], torch.cat((proxies[:index], proxies[(index+1):]))))
    #     angles.append(torch.min(angle))
    #     # angles.append(angle)
    # angles_ = torch.tensor(angles)
    # # angles_ = torch.cat(angles).squeeze()

    angles = torch.acos(pairwise_similarity(proxies, proxies)).cuda()
    angles_ = angles[torch.triu(torch.ones(expInfo.class_nb, expInfo.class_nb), 1) == 1]
    return angles_

def get_stretchedness_center(info_dict):
    """ Angles between global center and feature mean """
    theta = torch.acos(pairwise_similarity(info_dict['global center'], info_dict['feature mean'])).squeeze()

    return theta

def get_attachedness(expInfo, query_feature, query_label, proxies):
    angles = []
    for index in range(expInfo.class_nb):
        mask = (query_label == index)
        if expInfo.computer == '58':
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        angle = torch.acos(pairwise_similarity(query_feature[mask], proxies[index]))
        angles.append(angle)

    angles = torch.cat(angles).squeeze()
    return angles


def stretching_proxies(proxies, alpha0, eps, type):
    if type == '0':
        return proxies

    proxy_mean = normalize(torch.mean(proxies, dim=0))
    proxy_mean_norm = torch.norm(torch.mean(proxies, dim=0))
    proxy_theta = torch.acos(pairwise_similarity(proxy_mean, proxies)).squeeze()
    if type == '1':
        alpha = proxy_mean_norm * alpha0
        new_proxies = proxies * (torch.cos(alpha) + torch.sin(alpha) / torch.tan(proxy_theta))[:, None] - proxy_mean[None, :] * (torch.sin(alpha) / torch.sin(proxy_theta))[:, None]
    elif type == '2':
        alpha = proxy_mean_norm * alpha0 * (proxy_theta / torch.acos(torch.tensor(-1.0)))
        new_proxies = proxies * (torch.cos(alpha) + torch.sin(alpha) / torch.tan(proxy_theta))[:, None] - proxy_mean[None, :] * (torch.sin(alpha) / torch.sin(proxy_theta))[:, None]
    elif type == '3':
        direc = torch.zeros(proxies.size()).cuda()
        for index in range(len(proxies)):
            direc_indiv = proxies[index] - torch.cat((proxies[:index], proxies[(index + 1):]))
            direc_amp = torch.pow(torch.norm(direc_indiv, dim=1), 1)
            direc[index] = torch.mean(direc_indiv/direc_amp[:, None], dim=0)
        new_proxies = proxies + alpha0 * direc

    rand = eps * torch.randn(proxies.size())
    rand = rand.cuda()

    res = new_proxies + rand

    return l2_norm(res)

def rotating_matrix(x, y):
    n = len(x)
    x = normalize(x).cuda()
    y = normalize(y).cuda()
    theta = torch.acos(torch.min(torch.tensor(1.0).cuda(), torch.sum(x * y)))
    rotmat = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]).cuda()
    v = normalize(y - torch.cos(theta) * x)
    mat = torch.eye(n).cuda() - torch.mm(x.unsqueeze(1), x.unsqueeze(0))- torch.mm(v.unsqueeze(1), v.unsqueeze(0))
    aa = torch.mm(torch.mm(torch.stack((x, v)).t(), rotmat.cuda()), torch.stack((x, v)))

    return mat+aa

def set_proxies_globalcenter(feature_mean, lamb):
    global_mean = normalize(torch.mean(feature_mean, dim=0))
    # proxy_mean_norm = torch.norm(torch.mean(proxies, dim=0))
    proxy_theta = torch.acos(pairwise_similarity(global_mean, feature_mean)).squeeze()
    alpha = - (1.0 - lamb) * proxy_theta

    new_proxies = feature_mean * (torch.cos(alpha) + torch.sin(alpha) / torch.tan(proxy_theta))[:, None] - global_mean[None, :] * (torch.sin(alpha) / torch.sin(proxy_theta))[:, None]

    return new_proxies