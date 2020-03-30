# ------------------------------------------------------------------------------
# This code is base on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
from src.model.module.base_module import ConvModule

def BN(in_channel, only_activation=False, activation='relu'):
    if only_activation:
        return nn.ReLU(inplace=True)
    else:
        if activation == 'identity':
            return nn.BatchNorm2d(in_channel)
        else:
            return nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True, Norm=BN):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = Norm(out_dim) if with_bn else Norm(out_dim, only_activation=True)
        # self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        return bn
        # relu = self.relu(bn)
        # return relu

class residual(nn.Module):
    def __init__(self, inc, ouc, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = ConvModule(inc, ouc, k, stride=stride, padding=p)
        # self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        # self.bn1   = nn.BatchNorm2d(out_dim)
        # self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ConvModule(ouc, ouc, k, stride=1, padding=p, activation='linear')
        # self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        # self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip = ConvModule(inc, ouc, 1, stride=stride, activation='linear') if stride != 1 or inc != ouc else nn.Sequential()
        # self.skip  = nn.Sequential(
        #     nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
        #     nn.BatchNorm2d(out_dim)
        # ) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        skip  = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x + skip)

class fire_module(nn.Module):
    def __init__(self, inc, ouc, sr=2, stride=1):
        super(fire_module, self).__init__()
        self.conv1 = ConvModule(inc, ouc // sr, 1, activation='linear')
        # self.conv1    = nn.Conv2d(inp_dim, out_dim // sr, kernel_size=1, stride=1, bias=False)
        # self.bn1      = nn.BatchNorm2d(out_dim // sr)

        self.conv_1x1 = ConvModule(ouc // sr, ouc // 2, 1, stride=stride, activation='linear', use_bn=False)
        # self.conv_1x1 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=1, stride=stride, bias=False)
        self.conv_3x3 = ConvModule(ouc // sr, ouc // 2, 3, stride=stride, padding=1, activation='linear', use_bn=False)
        # self.conv_3x3 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=3, padding=1, 
                                #   stride=stride, groups=out_dim // sr, bias=False)
        self.skip = (stride == 1 and inc == ouc)
        self.bn  = nn.BatchNorm2d(ouc)        
        if self.skip:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_old = x
        x = self.conv1(x)
        x = torch.cat((self.conv_1x1(x), self.conv_3x3(x)), 1)
        x = self.bn(x)
        if self.skip:
            return self.relu(x + x_old)
        else:
            return x

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

def make_layer(inc, ouc, modules):
    layers  = [fire_module(inc, ouc)]
    layers += [fire_module(ouc, ouc) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def make_layer_revr(inc, ouc, modules):
    layers  = [fire_module(inc, inc) for _ in range(modules - 1)]
    layers += [fire_module(inc, ouc)]
    return nn.Sequential(*layers)

def make_hg_layer(inc, ouc, modules):
    layers  = [fire_module(inc, ouc, stride=2)]
    layers += [fire_module(ouc, ouc) for _ in range(1, modules)]
    return nn.Sequential(*layers)

class merge(nn.Module):
    def forward(self, x, y):
        return x + y

def make_merge_layer(dim):
    return merge()

class hg_module(nn.Module):
    def __init__(
        self, n, dims, modules, 
        make_up_layer,
        make_pool_layer, 
        make_hg_layer,
        make_low_layer, 
        make_hg_layer_revr,
        make_unpool_layer, 
        make_merge_layer,
        Norm=BN
    ):
        super(hg_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n    = n
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = hg_module(
            n - 1, dims[1:], modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer, 
            Norm=Norm
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2  = make_unpool_layer(curr_dim)
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        merg = self.merg(up1, up2)
        return merg

class HourglassNet(nn.Module):
    def __init__(self, stacks=2):
        super(HourglassNet, self).__init__()
        self.stacks = stacks
        # self.heads= heads

        self.pre     = nn.Sequential(
            ConvModule(3, 128, 7, stride=2, padding=3),
            # convolution(7, 3, 128, stride=2, Norm=Norm),
            residual(128, 256, stride=2),
            residual(256, 256, stride=2)
        )

        self.hg_mods = nn.ModuleList([
            hg_module(
                4, [256, 256, 384, 384, 512], [2, 2, 2, 2, 4],
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_up_layer=make_layer,
                make_low_layer=make_layer,
                make_hg_layer_revr=make_layer_revr,
                make_hg_layer=make_hg_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(stacks)
        ])

        self.cnvs    = nn.ModuleList([ConvModule(256, 256, 3, padding=1) for _ in range(stacks)])
        # self.cnvs    = nn.ModuleList([convolution(3, 256, 256, Norm=Norm) for _ in range(stacks)])
        self.inters  = nn.ModuleList([residual(256, 256) for _ in range(stacks - 1)])
        self.cnvs_   = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])
        self.inters_ = nn.ModuleList([self._merge_mod() for _ in range(stacks - 1)])    

        # for head in heads.keys():
        #     if 'hm' in head:
        #         module =  nn.ModuleList([
        #             self._pred_mod(heads[head]) for _ in range(stacks)
        #         ])
        #         self.__setattr__(head, module)
        #         for heat in self.__getattr__(head):
        #             heat[-1].bias.data.fill_(-2.19)
        #     else:
        #         module = nn.ModuleList([
        #             self._pred_mod(heads[head]) for _ in range(stacks)
        #         ])
        #         self.__setattr__(head, module)

        # self.relu = nn.LeakyReLU(inplace=True) if Norm is ABN else nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self._init_params()    

    def forward(self, image):
        inter = self.pre(image)
        outs  = []

        for ind in range(self.stacks):
            kp_, cnv_  = self.hg_mods[ind], self.cnvs[ind]
            kp  = kp_(inter)
            cnv = cnv_(kp)

            # out = {}
            # for head in self.heads:
            #     layer = self.__getattr__(head)[ind]
            #     y = layer(cnv)
            #     out[head] = y
            
            outs.append(cnv)
            if ind < self.stacks - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs
    
    def _merge_mod(self):
        return ConvModule(256, 256, 1, activation='linear')
        # return nn.Sequential(
        #     nn.Conv2d(256, 256, (1, 1), bias=False),
        #     nn.BatchNorm2d(256)
        # )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def hourglass_net(**kwargs):
    model = HourglassNet()
    return model

if __name__ == "__main__":
    model = hourglass_net()
    num = 0.0
    for p in model.parameters():
        num += p.numel()

    print(num / 1e6)

    x = torch.rand(2,3,512,512)
    output = model(x)