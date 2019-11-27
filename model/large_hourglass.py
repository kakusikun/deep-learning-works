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

from inplace_abn import InPlaceABN as IABN

def BN(in_channel, only_activation=False, activation='relu'):
    if only_activation:
        return nn.ReLU(inplace=True)
    else:
        if activation == 'identity':
            return nn.BatchNorm2d(in_channel)
        else:
            return nn.Sequential(nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True))

def ABN(in_channel, only_activation=False, activation='leaky_relu'):
    if only_activation:
        if activation == 'leaky_relu':
            return nn.LeakyReLU(inplace=True)
    else:
        return IABN(in_channel, activation=activation)

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
    def __init__(self, inp_dim, out_dim, k=3, stride=1, Norm=BN):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1 = Norm(out_dim)
        # self.bn1   = nn.BatchNorm2d(out_dim)
        # self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2 = Norm(out_dim, activation='identity')
        # self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            Norm(out_dim, activation='identity')
            # nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.LeakyReLU(inplace=True) if Norm is ABN else nn.ReLU(inplace=True)
        # self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        # relu1 = self.relu1(bn1)

        # conv2 = self.conv2(relu1)
        conv2 = self.conv2(bn1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class fire_module(nn.Module):
    def __init__(self, inp_dim, out_dim, sr=2, stride=1, Norm=BN):
        super(fire_module, self).__init__()
        self.conv1    = nn.Conv2d(inp_dim, out_dim // sr, kernel_size=1, stride=1, bias=False)
        self.bn1      = Norm(out_dim // sr, activation='identity')
        # self.bn1      = nn.BatchNorm2d(out_dim // sr)
        self.conv_1x1 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=1, stride=stride, bias=False)
        self.conv_3x3 = nn.Conv2d(out_dim // sr, out_dim // 2, kernel_size=3, padding=1, 
                                  stride=stride, groups=out_dim // sr, bias=False)
        # self.bn2      = nn.BatchNorm2d(out_dim)
        self.bn2      = Norm(out_dim, activation='identity')
        self.skip     = (stride == 1 and inp_dim == out_dim)
        self.relu     = nn.LeakyReLU(inplace=True) if Norm is ABN else nn.ReLU(inplace=True)
        # self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        conv2 = torch.cat((self.conv_1x1(bn1), self.conv_3x3(bn1)), 1)
        bn2   = self.bn2(conv2)
        if self.skip:
            return self.relu(bn2 + x)
        else:
            return self.relu(bn2)

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1)

def make_layer(inp_dim, out_dim, modules, Norm=BN):
    layers  = [fire_module(inp_dim, out_dim, Norm=Norm)]
    layers += [fire_module(out_dim, out_dim, Norm=Norm) for _ in range(1, modules)]
    return nn.Sequential(*layers)

def make_layer_revr(inp_dim, out_dim, modules, Norm=BN):
    layers  = [fire_module(inp_dim, inp_dim, Norm=Norm) for _ in range(modules - 1)]
    layers += [fire_module(inp_dim, out_dim, Norm=Norm)]
    return nn.Sequential(*layers)

def make_hg_layer(inp_dim, out_dim, modules, Norm=BN):
    layers  = [fire_module(inp_dim, out_dim, stride=2, Norm=Norm)]
    layers += [fire_module(out_dim, out_dim, Norm=Norm) for _ in range(1, modules)]
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
        self.up1  = make_up_layer(curr_dim, curr_dim, curr_mod, Norm=Norm)
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod, Norm=Norm)
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
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod, Norm=Norm)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod, Norm=Norm)
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
    def _pred_mod(self, dim, Norm=BN):
        return nn.Sequential(
            convolution(1, 256, 256, with_bn=False, Norm=Norm),
            nn.Conv2d(256, dim, (1, 1))
        )

    def _merge_mod(self, Norm=BN):
        return nn.Sequential(
            nn.Conv2d(256, 256, (1, 1), bias=False),
            Norm(256, activation='identity')
            # nn.BatchNorm2d(256)
        )

    def __init__(self, cfg, heads, stacks=2, Norm=BN):
        super(HourglassNet, self).__init__()
        self.stacks = stacks
        self.heads= heads

        self.pre     = nn.Sequential(
            convolution(7, 3, 128, stride=2, Norm=Norm),
            residual(128, 256, stride=2, Norm=Norm),
            residual(256, 256, stride=2, Norm=Norm)
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
                make_merge_layer=make_merge_layer,
                Norm=Norm
            ) for _ in range(stacks)
        ])

        self.cnvs    = nn.ModuleList([convolution(3, 256, 256, Norm=Norm) for _ in range(stacks)])
        self.inters  = nn.ModuleList([residual(256, 256, Norm=Norm) for _ in range(stacks - 1)])
        self.cnvs_   = nn.ModuleList([self._merge_mod(Norm=Norm) for _ in range(stacks - 1)])
        self.inters_ = nn.ModuleList([self._merge_mod(Norm=Norm) for _ in range(stacks - 1)])    

        self._init_params()    

        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    self._pred_mod(heads[head]) for _ in range(stacks)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    self._pred_mod(heads[head]) for _ in range(stacks)
                ])
                self.__setattr__(head, module)

        self.relu = nn.LeakyReLU(inplace=True) if Norm is ABN else nn.ReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=True)

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

    def forward(self, image):
        inter = self.pre(image)
        outs  = []

        for ind in range(self.stacks):
            kp_, cnv_  = self.hg_mods[ind], self.cnvs[ind]
            kp  = kp_(inter)
            cnv = cnv_(kp)

            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[ind]
                y = layer(cnv)
                out[head] = y
            
            outs.append(out)
            if ind < self.stacks - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

def get_large_hourglass_net(cfg):
    if cfg.TASK == 'object':
            heads = {'hm': cfg.MODEL.NUM_CLASSES,
                     'wh': 2,
                     'reg': 2}
    elif cfg.TASK == 'keypoint':
        heads = {'hm': cfg.MODEL.NUM_CLASSES, 
                    'wh': 2, 
                    'hps': 294*2,
                    'reg': 2,
                    'hm_hp': 294,
                    'hp_offset': 2}
    if cfg.MODEL.NORM == 'ABN':
        Norm = ABN
    else:
        Norm = BN
    model = HourglassNet(cfg, heads, Norm=Norm)
    return model
