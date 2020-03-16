from sys import maxsize
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.model.module.spos_modules import ShuffleNasBlock
from src.model.module.base_module import ConvModule

class ShuffleNetOneShot(nn.Module):
    def __init__(self,
        strides,
        stage_repeats,
        stage_out_channels,
        mode='plus'):
        super(ShuffleNetOneShot, self).__init__()

        channel_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.strides = strides
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels
        
        stemc = block_inc = self.stage_out_channels[0]
        self.stem = ConvModule(3, stemc, 3, stride=self.strides[0], padding=1, activation='hs', affine=False)

        self.max_pool = None
        if mode == 'v2':
            if strides[1] == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
            else:
                self.max_pool = nn.BatchNorm2d(block_inc)

        stages = []
        block_idx = 0
        for stage_i in range(len(self.stage_repeats)):
            num_blocks = self.stage_repeats[stage_i]
            ouc = self.stage_out_channels[stage_i+1]
            activation = 'hs' if stage_i >= 1 and mode == 'plus' else 'relu'
            useSE = 'True' if stage_i >= 2 and mode == 'plus' else False

            for i in range(num_blocks):
                if i == 0:
                    inc, stride = block_inc, self.strides[stage_i+1]
                else:
                    inc, stride = ouc, 1
                
                stages.append(ShuffleNasBlock(
                    inc,
                    ouc,
                    stride=stride,
                    channel_scales=channel_scales,
                    use_se=useSE,
                    act_name=activation))
                block_idx += 1
            block_inc = ouc

        self.stages = nn.ModuleList(stages)

        self._initialize_weights()

    def forward(self, x, block_choices, channel_choices):
        x = self.stem(x)
        if self.max_pool:
            x = self.max_pool(x)
        block_idx = 0
        for m in self.stages:
            x = m(x, block_choices[block_idx], channel_choices[block_idx])
            block_idx += 1
        assert block_idx == len(block_choices)
        return x

    def _get_flops_params(self, x):
        #TODO

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'stem' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

