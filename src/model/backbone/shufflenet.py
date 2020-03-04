import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.module.base_module import (
    ConvModule,
    SEModule,
    HSwish,
)

class ShuffleBlock(nn.Module):
    def __init__(self, inc, midc, ouc, ksize, stride, activation, useSE, mode):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        pad = ksize // 2
        inc = inc // 2 if stride == 1 else inc

        if mode == 'v2':
            branch_main = [
                ConvModule(inc, midc, 1, activation=activation),
                ConvModule(midc, midc, ksize, stride=stride, padding=pad, groups=midc, activation='linear'),
                ConvModule(midc, ouc - inc, 1, activation=activation),
            ]
        elif mode == 'xception':
            assert ksize == 3
            branch_main = [
                ConvModule(inc, inc, 3, stride=stride, padding=1, groups=inc, activation='linear'),
                ConvModule(inc, midc, 1, activation=activation),
                ConvModule(midc, midc, 3, stride=stride, padding=1, groups=midc, activation='linear'),
                ConvModule(midc, midc, 1, activation=activation),
                ConvModule(midc, midc, 3, stride=stride, padding=1, groups=midc, activation='linear'),
                ConvModule(midc, ouc - inc, 1, activation=activation),
            ]
        else:
            raise TypeError
        
        if activation == 'relu':
            assert useSE == False
        else:
            if useSE:
                branch_main.append(SEModule(ouc - inc))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            self.branch_proj = nn.Sequential(
                ConvModule(inc, inc, ksize, stride=stride, padding=pad, groups=inc, activation='linear'),
                ConvModule(inc, inc, 1, activation=activation),
            )
        else:
            self.branch_proj = None

    def forward(self, x):
        if self.stride==1:
            x_proj, x = channel_shuffle(x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class ShuffleNetV2_Plus(nn.Module):
    def __init__(self, block_choice=None, channel_choice=None, model_size='Large'):
        super(ShuffleNetV2_Plus, self).__init__()
        assert block_choice is not None
        assert channel_choice is not None

        channel_scale = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        self.stage_repeats = [4, 4, 8, 4]

        if model_size == 'Large':
            self.stage_out_channels = [68, 168, 336, 672]
        elif model_size == 'Medium':
            self.stage_out_channels = [48, 128, 256, 512]
        elif model_size == 'Small':
            self.stage_out_channels = [36, 104, 208, 416]
        else:
            raise TypeError
        
        stemc = block_inc = 16
        self.stem = ConvModule(3, stemc, 3, stride=2, padding=1, activation='hs')
        self.stages = nn.ModuleList()

        block_idx = 0
        for stage_i in range(len(self.stage_repeats)):
            stage = []
            num_blocks = self.stage_repeats[stage_i]
            ouc = self.stage_out_channels[stage_i]

            activation = 'hs' if stage_i >= 1 else 'relu'
            useSE = 'True' if stage_i >= 2 else False

            for i in range(num_blocks):
                if i == 0:
                    inc, stride = block_inc, 2
                else:
                    inc, stride = ouc, 1
                
                midc = make_divisible(int(ouc // 2 * channel_scale[channel_choice[block_idx]]))
                block_idx += 1

                if block_choice[block_idx] == 0:
                    stage.append(ShuffleBlock(inc, ouc, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE, mode='v2'), midc=midc)
                elif block_choice[block_idx] == 1:
                    stage.append(ShuffleBlock(inc, ouc, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE, mode='v2'), midc=midc)
                elif block_choice[block_idx] == 2:
                    stage.append(ShuffleBlock(inc, ouc, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE, mode='v2'), midc=midc)
                elif block_choice[block_idx] == 3:
                    stage.append(ShuffleBlock(inc, ouc, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE, mode='xception'), midc=midc)
                else:
                    raise TypeError
            block_inc = ouc
            self.stages.append(nn.Sequential(*stage))
        assert block_idx == len(block_choice)

        # TODO: move to head module
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#
        # self.conv_last = ConvModule(block_inc, featc, 1, activation='hs')
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.LastSE = SEModule(featc)
        # self.fc = nn.Sequential(
        #     nn.Linear(featc, featc, bias=False),
        #     HSwish(),
        # )
        # self.dropout = nn.Dropout(0.2)
        # self.classifier = nn.Sequential(nn.Linear(1280, n_class, bias=False))
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        stage_feats = []
        for stage in self.stages:
            x = stage(x)
            stage_feats.append(x)
        return stage_feats

        # TODO: move to head module
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#
        # x = self.conv_last(x)
        # x = self.gap(x)
        # x = self.LastSE(x)
        # x = x.contiguous().view(-1, 1280)
        # x = self.fc(x)
        # x = self.dropout(x)
        # x = self.classifier(x)
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

def shufflenetv2_plus(shufflenetv2_plus_model_size):
    block_choice = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    channel_choice = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    model = ShuffleNetV2_Plus(block_choice=block_choice, channel_choice=channel_choice, model_size='Small')
    return model

if __name__ == "__main__":
    block_choice = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    channel_choice = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    model = ShuffleNetV2_Plus(block_choice=block_choice, channel_choice=channel_choice, model_size='Small')
    num = 0.0
    for p in model.parameters():
        num += p.numel()

    print(num / 1e6)