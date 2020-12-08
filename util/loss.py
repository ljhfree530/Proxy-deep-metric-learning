# encoding: utf-8
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
# from pytorch_metric_learning import miners, losses

def topk_mask(input, dim, K = 10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim = dim, **kwargs)[1]
    return torch.autograd.Variable(torch.zeros_like(input.data)).scatter(dim, index, 1.0)

def pdist(A, squared = False, eps = 1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    return res if squared else res.clamp(min = eps).sqrt()


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def uclidean_dist_mat(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def pairwise_innerprod(x, y=None, getMean=False):
    if y is None:
        y = x
    if len(y.size()) == 1:
        y = y.unsqueeze(0)
    if len(x.size()) == 1:
        x = x.unsqueeze(0)
    n = x.size(0)
    # similarity
    innerprod = torch.mm(x, y.t())
    if getMean == True:
        return torch.trace(innerprod) / n
    return innerprod

def pairwise_similarity(x, y=None, getMean=False):
    if y is None:
        y = x
    # normalization
    if len(y.size()) == 1:
        y = y.unsqueeze(0)
    if len(x.size()) == 1:
        x = x.unsqueeze(0)
    y = normalize(y)
    x = normalize(x)
    if len(y.size()) == 1:
        y = y.unsqueeze(0)
    if len(x.size()) == 1:
        x = x.unsqueeze(0)
    y = y.cuda()
    x = x.cuda()
    n = x.size(0)
    # similarity
    similarity = torch.mm(x, y.t())

    if getMean == True:
        return torch.trace(similarity) / n

    return similarity

def potential(x, alpha=1.0):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
    Returns:
      pot: pytorch Variable
    """
    pot = 0.0
    n = x.size(0)
    # x = normalize(x)


    if n == 1:
        return 0
    for i in range(n-1):
        y1 = x[i, :].unsqueeze(0)
        y2 = x[(i+1):,:]
        dists = euclidean_dist(y1, y2)
        pot += torch.pow(dists, -alpha).sum()

    return pot

def potential_normalized(x, alpha=1.0):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
    Returns:
      pot: pytorch Variable
    """
    pot = 0.0
    n = x.size(0)
    eps = 1e-6

    sim = pairwise_similarity(x)
    dists = torch.pow(2.0 - 2.0 * sim + eps, 0.5)
    dists[range(n), range(n)] = 1.0
    pot = ( torch.pow(dists, -alpha).sum() - n ) / 2.0

    return pot

def potential_normalized_softmax(x, l = 1.0):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
    Returns:
      pot: pytorch Variable
    """
    pot = 0.0
    n = x.size(0)
    dists_vec = torch.zeros(int(n*(n-1)/2))
    eps = 1e-6

    sim = pairwise_similarity(x)
    # dists = torch.pow(2.0 - 2.0 * sim + eps, 0.5)

    sims_vec = sim[torch.triu(torch.ones(n, n), 1) == 1]

    soft = torch.softmax(l * sims_vec, dim=0)
    entropy = Categorical(probs = soft).entropy()

    return sims_vec, entropy

def smooth_max(x, alpha = 10.0, ind = 1):
    exp = torch.exp(alpha * x)
    prob = exp / exp.sum()
    max_ = torch.mul(x, prob).sum()

    return max_



def potential_normalized_exp(x, alpha=1.0):
    """
    Args:
      x: pytorch Variable, with shape [n, d]
    Returns:
      pot: pytorch Variable
    """
    pot = 0.0
    n = x.size(0)
    eps = 1e-6

    sim = pairwise_similarity(x)
    dists = torch.pow(2.0 - 2.0 * sim + eps, 0.5)
    dists[range(n), range(n)] = 1.0
    pot = ( torch.exp(torch.pow(dists, -alpha)).sum() ) / 2.0

    return pot





def hard_example_mining(dist_mat, labels, margin, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    torch.set_printoptions(threshold=5000) 
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, f1, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, self.margin)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss #, dist_ap, dist_an




# class TripletLoss_new(nn.Module):
#     def __init__(self, margin=0.1, **kwargs):
#         super(TripletLoss_new, self).__init__()
#         self.margin = margin
#         self.miner = miners.TripletMarginMiner(margin, type_of_triplets='all')
#         self.loss_func = losses.TripletMarginLoss(margin=self.margin)
#
#     def forward(self, embeddings, labels):
#         hard_pairs = self.miner(embeddings, labels)
#         loss = self.loss_func(embeddings, labels, hard_pairs)
#         return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class CompactnessLossTheta:
    def __init__(self, margin=None, pc_direct=False):
        self.margin = margin
        self.pc_direct = pc_direct

    def __call__(self, f1, f2, labels, normalize_feature=False):
        f1 = f1.detach()
        s1_mat = pairwise_similarity(f1)
        s2_mat = pairwise_similarity(f2)
        Pscore = 0.0
        Pcount = 0

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i > j:
                    # Positivie
                    if labels[i] == labels[j]:
                        if self.pc_direct:
                            temp = 1 - s2_mat[i, j]
                        else:
                            temp = s2_mat[i, j] - s1_mat[i, j] + self.margin

                        if temp > 0:
                            Pscore = Pscore + temp
                            Pcount += 1

        if not Pcount == 0:
            lossPE = Pscore / Pcount
        else:
            lossPE = 0

        return lossPE

class ContrastiveLossTheta:
    def __init__(self, margin=None, pc_direct=False):
        self.margin = margin
        self.pc_direct = pc_direct

    def __call__(self, f1, f2, labels, normalize_feature=False):
        f1 = f1.detach()
        s1_mat = pairwise_similarity(f1)
        s2_mat = pairwise_similarity(f2)
        Pscore = 0.0
        Pcount = 0
        Nscore = 0.0
        Ncount = 0

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i > j:
                    # Positivie
                    if labels[i] == labels[j]:
                        if self.pc_direct:
                            temp = 1 - s2_mat[i, j]
                        else:
                            temp = s2_mat[i, j] - s1_mat[i, j] + self.margin/10

                        if temp > 0:
                            Pscore = Pscore + temp
                            Pcount += 1
                    else:
                        if self.pc_direct:
                            temp = s2_mat[i, j]
                        else:
                            temp = s1_mat[i, j] - s2_mat[i, j] + self.margin/10
                        if temp > 0:
                            Nscore = Nscore + temp
                            Ncount += 1

        if not Pcount == 0:
            lossPE = Pscore / Pcount
        else:
            lossPE = 0

        if not Ncount == 0:
            lossNE = Nscore / Ncount
        else:
            lossNE = 0

        return 10*lossPE + lossNE


class CompactnessLoss:
    def __init__(self, margin=None, pc_direct=False):
        self.margin = margin
        self.pc_direct = pc_direct

    def __call__(self, f1, f2, labels, normalize_feature=False):
        f1 = f1.detach()
        d1_mat = uclidean_dist_mat(f1)
        d2_mat = uclidean_dist_mat(f2)
        Pscore = 0.0
        Pcount = 0

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i > j:
                    # Positivie
                    if labels[i] == labels[j]:
                        if self.pc_direct:
                            temp = d2_mat[i, j] ** 0.5
                        else:
                            temp = d2_mat[i, j] ** 0.5 - d1_mat[i, j] ** 0.5 + self.margin

                        if temp > 0:
                            Pscore = Pscore + temp
                            Pcount += 1

        if not Pcount == 0:
            lossPE = Pscore / Pcount
        else:
            lossPE = 0

        return lossPE

class HLoss:
    def __init__(self, reg=0.001, alpha=1):
        self.reg = reg
        self.alpha = alpha

    def __call__(self, x):
        return self.reg * potential(x, self.alpha)

class HLoss_normalized:
    def __init__(self, reg=0.001, alpha=1):
        self.reg = reg
        self.alpha = alpha

    def __call__(self, x):
        return self.reg * potential_normalized(x, self.alpha)

class EntropyLoss:
    def __init__(self, margin=None, pc_direct=False):
        self.margin = margin
        self.pc_direct = pc_direct

    def __call__(self, f1, f2, labels, normalize_feature=False):
        f1 = f1.detach()
        d1_mat = uclidean_dist_mat(f1)
        d2_mat = uclidean_dist_mat(f2)
        Pscore = 0.0
        Pcount = 0

        for i in range(len(labels)):
            for j in range(len(labels)):
                if i > j:
                    # Positivie
                    if labels[i] == labels[j]:
                        if self.pc_direct:
                            temp = d2_mat[i, j] ** 0.5
                        else:
                            temp = d2_mat[i, j] ** 0.5 - d1_mat[i, j] ** 0.5 + self.margin

                        if temp > 0:
                            Pscore = Pscore + temp
                            Pcount += 1

        if not Pcount == 0:
            lossPE = Pscore / Pcount
        else:
            lossPE = 0

        return lossPE

class Margin:
    def __call__(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        alpha = 0.2
        beta = 1.2
        distance_threshold = 0.5
        inf = 1e6
        eps = 1e-6
        distance_weighted_sampling = True
        d = pdist(embeddings)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.autograd.Variable(torch.eye(len(d))).type_as(d)
        num_neg = int(pos.data.sum() / len(pos))
        if distance_weighted_sampling:
            '''
            dim = embeddings.size(-1)
            distance = d.data.clamp(min = distance_threshold)
            distribution = distance.pow(dim - 2) * ((1 - distance.pow(2) / 4).pow(0.5 * (dim - 3)))
            weights = distribution.reciprocal().masked_fill_(pos.data + torch.eye(len(d)).type_as(d.data) > 0, eps)
            samples = torch.multinomial(weights, replacement = False, num_samples = num_neg)
            neg = torch.autograd.Variable(torch.zeros_like(pos.data).scatter_(1, samples, 1))
            '''
            neg = torch.autograd.Variable(torch.zeros_like(pos.data).scatter_(1, torch.multinomial((d.data.clamp(min = distance_threshold).pow(embeddings.size(-1) - 2) * (1 - d.data.clamp(min = distance_threshold).pow(2) / 4).pow(0.5 * (embeddings.size(-1) - 3))).reciprocal().masked_fill_(pos.data + torch.eye(len(d)).type_as(d.data) > 0, eps), replacement = False, num_samples = num_neg), 1))
        else:
            neg = topk_mask(d  + inf * ((pos > 0) + (d < distance_threshold)).type_as(d), dim = 1, largest = False, K = num_neg)
        L = F.relu(alpha + (pos * 2 - 1) * (d - beta))
        M = ((pos + neg > 0) * (L > 0)).float()
        return (M * L).sum() / M.sum(), 0


class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=10, beta=2, margin=0.5, hard_mining=None, **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, embeddings, labels):
        '''
        score = embeddings
        target = labels
        loss = 0
        counter = 0
        bsz = score.size(0)
        mag = (score ** 2).sum(1).expand(bsz, bsz)
        sim = score.mm(score.transpose(0, 1))
        dist = (mag + mag.transpose(0, 1) - 2 * sim)
        dist = torch.nn.functional.relu(dist).sqrt()

        for i in range(bsz):
            t_i = target[i].item()
            for j in range(i + 1, bsz):
                t_j = target[j].item()
                if t_i == t_j:
                    # Negative component
                    # !! Could do other things (like softmax that weights closer negatives)
                    l_ni = (self.margin - dist[i][target != t_i]).exp().sum()
                    l_nj = (self.margin - dist[j][target != t_j]).exp().sum()
                    l_n  = (l_ni + l_nj).log()
                    # Positive component
                    l_p  = dist[i,j]
                    loss += torch.nn.functional.relu(l_n + l_p) ** 2
                    counter += 1
        return loss / (2 * counter), 0
        '''
        margin = 1.0
        eps = 1e-4
        d = pdist(embeddings, squared=False, eps=eps)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d)
        neg_i = torch.mul((margin - d).exp(), 1 - pos).sum(1).expand_as(d)
        return torch.sum(F.relu(pos.triu(1) * ((neg_i + neg_i.t()).log() + d)).pow(2)) / (pos.sum() - len(d)), 0