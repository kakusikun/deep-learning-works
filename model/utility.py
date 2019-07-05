import sys
import math
import glog

import torch
import torch.nn as nn
import torch.nn.functional as F

from solver.solvers import *

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

class FC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(FC, self).__init__()
        glog.check_gt(num_classes, 0)
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.fc(x)

class CrossEntropyLossLSR(nn.Module):
    def __init__(self, num_classes, eps=0.1):
        super(CrossEntropyLossLSR, self).__init__()
        glog.check_gt(num_classes, 0)
        self.eps = eps
        self.prior = eps/num_classes

    def forward(self, inputs, labels):
        device = inputs.get_device()

        target = labels.view(-1,1).long()
        p = torch.ones(inputs.size()) * self.prior
        if device > -1:
            target = target.to(device)
            p = p.to(device)

        p.scatter_(1, target, (1 - self.eps + self.prior))
        log_q = F.log_softmax(inputs, 1)

        cross_entropy = -1 * (p * log_q).sum(1).mean()

        return cross_entropy

class CenterPushLoss(nn.Module):
    r"""Implement of global push, center, push loss in RMNet: :
    Args:
        in_features: size of features
        num_classes: number of identity in dataset
        K: number of image per identity
        m: margin
        center, global push, push with size N, batch size
    """
    def __init__(self, in_features, num_classes, m=0.50, K=8):
        super(CenterPushLoss, self).__init__()
        self.m = m
        self.K = K
        self.center = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        self.ranking_loss = nn.MarginRankingLoss(margin=self.m, reduction="none")
        nn.init.xavier_uniform_(self.center)
        

    def forward(self, inputs, labels):
        device = inputs.get_device()
        n = inputs.size(0)
        m = self.center.size(0)  
        center_feature = F.normalize(self.center)  

        cdist = F.linear(inputs, center_feature)         
        # cdist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, m) + \
        #       torch.pow(center_feature, 2).sum(dim=1, keepdim=True).expand(m ,n).t()
        # cdist.addmm_(1, -2, inputs, center_feature.t())
        # cdist = cdist.clamp(min=1e-12).sqrt()   
        
        dist = F.linear(inputs, inputs)
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())        
        # dist = dist.clamp(min=1e-12).sqrt()              

        target = labels.view(-1,1).long()
        p = torch.zeros(cdist.size())
        cy = torch.ones(m-1)
        y = torch.ones(n-self.K)

        if device > -1:
            target = target.to(device)
            p = p.to(device)
            cy = cy.to(device)
            y = y.to(device)

        p.scatter_(1, target, 1)
        cdist_p = cdist[p==1].expand(m-1, n).t()
        cdist_n = cdist[p==0].reshape(n, -1)       
        gpush = self.ranking_loss(cdist_p, cdist_n, cy).mean(1)      

        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_p = cdist[p==1].expand(n-self.K, n).t()
        dist_n = dist[mask==0].reshape(n, -1)        
        push = self.ranking_loss(dist_p, dist_n, y).mean(1)

        center = 1.0 - cdist[p==1]
        
        return center, gpush, push

class AMCrossEntropyLossLSR(nn.Module):
    r"""Implement of large margin cosine distance in cross entropy with label smoothing: :
    Args:
        in_features: size of each input sample
        num_classes: number of identity in dataset
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, num_classes, s=30.0, m=0.40, eps=0.1):
        super(AMCrossEntropyLossLSR, self).__init__()
        self.s = s
        self.m = m
        self.eps = eps
        self.prior = eps/num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, labels):
        device = inputs.get_device()

        cosine = F.linear(inputs, F.normalize(self.weight))
        phi = cosine - self.m

        one_hot = torch.zeros(cosine.size())
        target = labels.view(-1,1).long()
        p = torch.ones(cosine.size()) * self.prior

        if device > -1:
            one_hot = one_hot.to(device)
            target = target.to(device)
            p = p.to(device)
           
        p.scatter_(1, target, (1 - self.eps + self.prior))
        one_hot.scatter_(1, target, 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        log_q = F.log_softmax(output, 1)
        
        cross_entropy = -1 * (p * log_q).sum(1)
        return cross_entropy

class FocalWeight():
    def __init__(self, length, batch_size, delta=0.16, alpha=0.25, gamma=2.0, device=0):
        self.a = alpha
        self.r = gamma
        self.d = delta
        self.bs = batch_size
        self.lg = length
        # 1st column -> previous
        # 2nd column -> current
        self.k = torch.zeros(length, 2)  

        self.change_target = False

        if device > 0:
            self.k = self.k.cuda()
    
    def get_weighted_losses(self, loss):

        self.k[:,0] = self.k[:,1]

        self.k[:,1] = self.a * loss.detach() + (1 - self.a) * self.k[:,0]

        p = self.k.min(dim=1)[0] / self.k[:,0]

        focal_weight = -1 * torch.pow(1 - p, self.r) * torch.log(p)

        loss = focal_weight * loss     

        self.change_target = (focal_weight[1] / focal_weight[0]) < self.d

        return loss
    
    def weight_initialize(self, cores, data, use_gpu=False):
        glog.info("FocalWeight initialize ...")
        batch = next(iter(data))
        images, labels, _ = batch
        if use_gpu:
            images, labels = images.cuda(), labels.cuda()
        
        local, glob = cores['main'](images)

        local_loss = list(cores['local_loss'](local, labels))
        glob_loss = cores['glob_loss'](glob, labels)
        loss = torch.stack([glob_loss] + local_loss)    
        lg, bs = loss.size()
        _, indice = loss[:3].sum(0).sort(descending=True)

        if use_gpu:
            mask = torch.zeros(loss.size(1)).cuda()
        else:
            mask = torch.zeros(loss.size(1))

        effective_idx = mask.scatter(0, indice[:bs//2], 1).expand_as(loss)  
        loss = loss[effective_idx == 1].view(lg, bs//2).mean(1)

        glob_loss = loss[0]
        
        local_loss = loss[1:].sum()    

        loss = torch.cat([glob_loss, local_loss])

        self.k[:,1] = loss.detach()



class GradNorm():
    def __init__(self, cfg, shared_weight, alpha=0.16, device=-1):
        self.shared_weight = shared_weight
        self.alpha = alpha
        self.length = cfg.OPTIMIZER.NUM_LOSSES
        self.need_initial = True

        self.fc = nn.Linear(self.length, 1, bias=False)
        self.g_weights = self.fc.weight
        nn.init.constant_(self.g_weights, 1.0)
        
        self.l1_loss = nn.L1Loss(size_average=False)
        self.initial_loss = None
        self.opt = NovoGrad(self.fc.parameters(), amsgrad=True)

        if device != -1:
            self.fc = self.fc.cuda()
    @property
    def weights(self):
        self.g_weights.data = F.normalize(self.g_weights.detach(), p=1) * self.length        
        return self.g_weights.detach().squeeze()
    
    def loss_weight_backward(self, loss):
        w = self.g_weights.squeeze()
        GWt_norms = []
        for i in range(loss.size(0)):
            shared_weight_grad = torch.autograd.grad(loss[i],
                                                     self.shared_weight,
                                                     retain_graph=True)  
           
            GWt = w[i] * shared_weight_grad[0].detach()
            GWt_norms.append(GWt.norm())
        
        GWt_norms = torch.stack(GWt_norms)

        loss_ratios = loss.detach() / self.initial_loss
        inverse_training_rates = loss_ratios / loss_ratios.mean()

        desired_GWt_norm = GWt_norms.mean().detach() * torch.pow(inverse_training_rates, self.alpha)

        L_grad = self.l1_loss(GWt_norms, desired_GWt_norm)
        self.opt.zero_grad()
        L_grad.backward()
        self.opt.step()

    def weight_initialize(self, cores, data, use_gpu=False):
        glog.info("GradNorm initialize ...")
        losses = []
        for idx, batch in enumerate(data):
            if idx > 5:
                break
            images, labels, _ = batch
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            
            local, glob = cores['main'](images)

            local_loss = list(cores['local_loss'](local, labels))
            glob_loss = cores['glob_loss'](glob, labels)
            loss = torch.stack([glob_loss] + local_loss)
            losses.append(loss)
            glog.info(idx)
        
        esitmated_loss = torch.stack(losses).mean(0).detach()
        lg, bs = esitmated_loss.size()
        _, indice = esitmated_loss[:3].sum(0).sort(descending=True)
        if use_gpu:
            mask = torch.zeros(esitmated_loss.size(1)).cuda()
        else:
            mask = torch.zeros(esitmated_loss.size(1))

        effective_idx = mask.scatter(0, indice[:bs//2], 1).expand_as(esitmated_loss)   
        esitmated_loss = esitmated_loss[effective_idx == 1].view(lg, bs//2)
        self.initial_loss = esitmated_loss.mean(1)
        self.need_initial = False


if __name__ == '__main__':
    loss = CenterPushLoss(10, 10, m=0.5, K=2)
    amcellsr = AMCrossEntropyLossLSR(10, 10)
    loss = loss.cuda()
    amcellsr = amcellsr.cuda()
    inputs = torch.rand(10,10).cuda()
    targets = torch.LongTensor([1,1,2,2,3,3,7,7,6,6]).cuda()
    print(loss(inputs, targets))
    print(amcellsr(inputs, targets))



        





