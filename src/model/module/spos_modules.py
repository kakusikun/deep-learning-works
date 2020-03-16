import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model.module.base_module import (
    ConvModule,
    SEModule,
    HSwish,
)

from src.model.backbone.shufflenetv2_plus import ShuffleBlock

class ShuffleNetCSBlock(nn.Module):
    def __init__(self, 
        input_channel, 
        output_channel, 
        channel_scales, 
        ksize, 
        stride, 
        block_mode='v2', 
        act_name='relu', 
        use_se=False, 
        **kwargs):
        super(ShuffleNetCSBlock, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert block_mode in ['v2', 'xception']

        self.stride = stride
        self.ksize = ksize
        self.block_mode = block_mode
        self.input_channel = input_channel
        self.output_channel = output_channel
        """
        Regular block: (We usually have the down-sample block first, then followed by repeated regular blocks)
        Input[64] -> split two halves -> main branch: [32] --> mid_channels (final_output_C[64] // 2 * scale[1.4])
                        |                                       |--> main_out_C[32] (final_out_C (64) - input_C[32]
                        |
                        |-----> project branch: [32], do nothing on this half
        Concat two copies: [64 - 32] + [32] --> [64] for final output channel

        =====================================================================

        In "Single path one shot nas" paper, Channel Search is searching for the main branch intermediate #channel.
        And the mid channel is controlled / selected by the channel scales (0.2 ~ 2.0), calculated from:
            mid channel = block final output # channel // 2 * scale

        Since scale ~ (0, 2), this is guaranteed: main mid channel < final output channel
        """
        self.block = nn.ModuleList()
        for i in range(len(channel_scales)):
            mid_channel = int(output_channel // 2 * channel_scales[i])
            self.block.append(ShuffleBlock(
                self.input_channel, 
                mid_channel,
                self.output_channel, 
                ksize=ksize,
                stride=stride,
                activation=act_name,
                useSE=use_se,
                affine=False,
                mode=block_mode
            ))

    def forward(self, x, channel_choice):
        return self.block[channel_choice](x)

    def copy_weight(self):
        src_weight = self.block[-1].state_dict()
        for trt_module in self.block[:-1]:
            trt_weight = trt_module.state_dict()
            for w_name in trt_weight:
                if trt_weight[w_name].dim() > 0:
                    n_c = trt_weight[w_name].size(0)
                    trt_weight[w_name] = src_weight[w_name][:n_c,...]

class ShuffleNasBlock(nn.Module):
    def __init__(self, 
        input_channel, 
        output_channel, 
        stride,
        channel_scales, 
        act_name='relu', 
        use_se=False):
        super(ShuffleNasBlock, self).__init__()
        assert stride in [1, 2]
        """
        Four pre-defined blocks
        """
        self.nas_block = nn.ModuleList()

        self.nas_block.append(
            nn.Sequential()
        )
        self.nas_block.append(
            ShuffleNetCSBlock(input_channel, output_channel, channel_scales,
            3, stride, block_mode='v2', act_name=act_name, use_se=use_se)
        )
        self.nas_block.append(
            ShuffleNetCSBlock(input_channel, output_channel, channel_scales,
            3, stride, block_mode='xception', act_name=act_name, use_se=use_se)
        )

    def forward(self, x, block_choice, channel_choice):
        x = self.nas_block[block_choice](x, channel_choice)
        return x
