import sys
import math
import glog

import torch
import torch.nn as nn
import torch.nn.functional as F

# from solver.solvers import *

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

def hard_example_mining(dist_mat, labels, return_inds=False):
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


class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class ConvFC(nn.Module):
    def __init__(self, in_planes, out_planes, bias=False):
        super(ConvFC, self).__init__()
        self.fc = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(1,1), stride=(1,1), padding=(0,0), groups=1, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        return x

class CrossEntropyLossLS(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLossLS, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        device = inputs.get_device()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if device > -1: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class CenterLoss(nn.Module):
    r"""Implement of center loss: :
    Args:
        in_features: size of features
        num_classes: number of identity in dataset
    """
    def __init__(self, in_features, num_classes, use_gpu=True):
        super(CenterLoss, self).__init__()
        if use_gpu:
            self.center = nn.Parameter(torch.randn(num_classes, in_features).cuda())
        else:
            self.center = nn.Parameter(torch.randn(num_classes, in_features))

    def forward(self, inputs, labels):
        device = inputs.get_device()
            
        n = inputs.size(0)
        m = self.center.size(0)  

        #  cdist = F.linear(inputs, center_feature)         
        cdist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, m) + \
             torch.pow(self.center, 2).sum(dim=1, keepdim=True).expand(m ,n).t()
        cdist.addmm_(1, -2, inputs, self.center.t())        

        target = labels.view(-1,1).long()
        p = torch.zeros(cdist.size())

        if device > -1:
            target = target.to(device)
            p = p.to(device)

        p.scatter_(1, target, 1)

        center_loss = cdist[p==1].clamp(min = 1e-12, max = 1e+12).mean()
        
        return center_loss

# class PushLoss(nn.Module):
#     r"""Implement of global push, center, push loss in RMNet: :
#     Args:
#         in_features: size of features
#         num_classes: number of identity in dataset
#         K: number of image per identity
#         m: margin
#         center, global push, push with size N, batch size
#     """
#     def __init__(self, in_features, num_classes):
#         super(PushLoss, self).__init__()
#         self.center = nn.Parameter(torch.randn(num_classes, in_features))
#         #  nn.init.xavier_uniform_(self.center)        

#     def forward(self, inputs, labels):
#         device = inputs.get_device()
            
#         n = inputs.size(0)
#         m = self.center.size(0)  

#         dist_mat = euclidean_dist(inputs, inputs)  
#         c_dist_mat = euclidean_dist(inputs, normalize(self.center))  
              

#         target = labels.view(-1,1).long()
#         p = torch.zeros(cdist.size())

#         if device > -1:
#             target = target.to(device)
#             p = p.to(device)

#         p.scatter_(1, target, 1)

#         center_loss = cdist[p==1].clamp(min = 1e-12, max = 1e+12).mean()
        
#         return center_loss

class TupletLoss(nn.Module):
    r"""Implement of global push, center, push loss in RMNet: :
    Args:
        m: margin
        thresh: similarity for hard negative
    """
    def __init__(self, m=0.3, thresh=0.6):
        super(TupletLoss, self).__init__()
        self.m = m
        self.thresh = thresh

    def forward(self, inputs, labels):
        device = inputs.get_device()

        n = inputs.size(0)
       
        dist = F.linear(inputs, inputs)

        target = labels.view(-1,1).long()
        rank_mask = 1 - torch.eye(n)
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())

        if device > -1:
            target = target.to(device)
            rank_mask = rank_mask.to(device)
            
        true = mask[rank_mask==1].reshape(n,-1)
        rank1 = dist[rank_mask==1].reshape(n,-1).max(1)[1]
        match = true.gather(1, rank1.long().view(-1,1))

        push = []

        for i in range(n):
            dist_p = dist[i][mask[i]==1]
            dist_n = dist[i][mask[i]==0]  
            dist_n = dist_n[dist_n >= self.thresh]   
            rank = torch.exp(dist_n - dist_p.min() + self.m)
            push.append(torch.log(rank[rank > 1].sum() + 1) + 2 * torch.pow(dist_p - 1, 2).mean() / 2)

        push = torch.stack(push).mean()
        
        acc = match.float().mean()
        
        return push, acc.item()

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

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class AMSoftmax(nn.Module):
    r"""Implement of large margin cosine distance in cross entropy with label smoothing: :
    Args:
        in_features: size of each input sample
        num_classes: number of identity in dataset
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, num_classes, s=30.0, m=0.30):
        super(AMSoftmax, self).__init__()
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels):
        device = inputs.get_device()

        cosine = F.linear(inputs, F.normalize(self.weight))
        phi = cosine - self.m

        one_hot = torch.zeros(cosine.size())
        target = labels.view(-1,1).long()

        if device > -1:
            one_hot = one_hot.to(device)
            target = target.to(device)
           
        one_hot.scatter_(1, target, 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

if __name__ == '__main__':
    center_loss = CenterLoss(2048, 751)
    features = torch.ones(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()

    c_loss = center_loss(features, targets)
    print(c_loss)



        





