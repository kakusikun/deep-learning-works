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
    def __init__(self, inc, midc, ouc, ksize, stride, activation, useSE, mode, affine=True):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        pad = ksize // 2
        inc = inc // 2 if stride == 1 else inc

        if mode == 'v2':
            branch_main = [
                ConvModule(inc, midc, 1, activation=activation, affine=affine),
                ConvModule(midc, midc, ksize, stride=stride, padding=pad, groups=midc, activation='linear', affine=affine),
                ConvModule(midc, ouc - inc, 1, activation=activation, affine=affine),
            ]
        elif mode == 'xception':
            assert ksize == 3
            branch_main = [
                ConvModule(inc, inc, 3, stride=stride, padding=1, groups=inc, activation='linear', affine=affine),
                ConvModule(inc, midc, 1, activation=activation, affine=affine),
                ConvModule(midc, midc, 3, stride=1, padding=1, groups=midc, activation='linear', affine=affine),
                ConvModule(midc, midc, 1, activation=activation, affine=affine),
                ConvModule(midc, midc, 3, stride=1, padding=1, groups=midc, activation='linear', affine=affine),
                ConvModule(midc, ouc - inc, 1, activation=activation, affine=affine),
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
                ConvModule(inc, inc, ksize, stride=stride, padding=pad, groups=inc, activation='linear', affine=affine),
                ConvModule(inc, inc, 1, activation=activation, affine=affine),
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
    def __init__(self,
        strides,
        stage_repeats,
        stage_out_channels,
        block_choice=None, 
        channel_choice=None, 
        mode='plus'):
        super(ShuffleNetV2_Plus, self).__init__()
        assert block_choice is not None
        assert channel_choice is not None

        channel_scale = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.strides = strides
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels
        
        stemc = block_inc = self.stage_out_channels[0]
        self.stem = ConvModule(3, stemc, 3, stride=self.strides[0], padding=1, activation='hs')

        self.max_pool = None
        if mode == 'v2':
            if strides[1] == 2:
                self.max_pool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
            else:
                self.max_pool = nn.BatchNorm2d(block_inc)

        self.stages = nn.ModuleList()

        block_idx = 0
        for stage_i in range(len(self.stage_repeats)):
            stage = []
            num_blocks = self.stage_repeats[stage_i]
            ouc = self.stage_out_channels[stage_i+1]

            activation = 'hs' if stage_i >= 1 and mode == 'plus' else 'relu'
            useSE = 'True' if stage_i >= 2 and mode == 'plus' else False

            for i in range(num_blocks):
                if i == 0:
                    inc, stride = block_inc, self.strides[stage_i+1]
                else:
                    inc, stride = ouc, 1
                
                midc = int(ouc // 2 * channel_scale[channel_choice[block_idx]])

                if block_choice[block_idx] == 0:
                    stage.append(ShuffleBlock(inc, midc, ouc, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE, mode='v2'))
                elif block_choice[block_idx] == 1:
                    stage.append(ShuffleBlock(inc, midc, ouc, ksize=5, stride=stride,
                                    activation=activation, useSE=useSE, mode='v2'))
                elif block_choice[block_idx] == 2:
                    stage.append(ShuffleBlock(inc, midc, ouc, ksize=7, stride=stride,
                                    activation=activation, useSE=useSE, mode='v2'))
                elif block_choice[block_idx] == 3:
                    stage.append(ShuffleBlock(inc, midc, ouc, ksize=3, stride=stride,
                                    activation=activation, useSE=useSE, mode='xception'))
                else:
                    raise TypeError
                block_idx += 1
            block_inc = ouc
            self.stages.append(nn.Sequential(*stage))
        assert block_idx == len(block_choice)
        self.last_channel = block_inc

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)        
        stage_feats = []
        for stage in self.stages:
            x = stage(x)
            stage_feats.append(x)
        return stage_feats

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

def shufflenetv2_plus(model_size='Medium', **kwargs):
    block_choice = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    channel_choice = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    # block_choice = [1, 3, 2, 3, 3, 1, 2, 0, 3, 0, 2, 3, 0, 0, 1, 2, 2, 2, 3, 1] 
    # channel_choice = [8, 7, 5, 7, 1, 7, 7, 5, 1, 4, 0, 1, 0, 5, 1, 2, 3, 8, 2, 8]
    strides = [2, 2, 2, 2, 2]
    stage_repeats = [4, 4, 8, 4]

    if model_size == 'Large':
        stage_out_channels = [16, 68, 168, 336, 672]
    elif model_size == 'Medium':
        stage_out_channels = [16, 48, 128, 256, 512]
    elif model_size == 'Small':
        stage_out_channels = [16, 36, 104, 208, 416]
    elif model_size == 'OneShot':
        stage_out_channels = [16, 64, 160, 320, 640]
    else:
        raise TypeError

    model = ShuffleNetV2_Plus(
        strides=strides,
        stage_repeats=stage_repeats,
        stage_out_channels=stage_out_channels,
        block_choice=block_choice, 
        channel_choice=channel_choice)
    return model

def shufflenetv2(
        block_choice=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        channel_choice=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    ):
    model = ShuffleNetV2_Plus(
        strides=[1, 1, 2, 2, 2],
        stage_repeats=[4, 8, 4],
        stage_out_channels=[24, 116, 232, 464],
        block_choice=block_choice, 
        channel_choice=channel_choice,
        mode='v2')
    return model

if __name__ == "__main__":
    import torch
    import numpy as np
    import random

    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    np.random.seed(42)
    random.seed(42)

    # block_choice = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0]
    # channel_choice = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    # model = ShuffleNetV2_Plus(
    #     strides=[2,2,2],
    #     stage_repeats=[4,4,8],
    #     stage_out_channels=[64, 160, 320],
    #     block_choice=block_choice, channel_choice=channel_choice)
    # num = 0.0
    # for p in model.parameters():
    #     num += p.numel()

    # print(num / 1e6)
    model = shufflenetv2_plus()
    x = torch.ones(2,3,112,112)
    output = model(x)
    print(output)