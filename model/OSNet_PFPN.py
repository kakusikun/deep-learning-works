from __future__ import absolute_import
from __future__ import division

__all__ = ['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0']

import sys
import copy
import torch
from torch import nn
from torch.nn import functional as F
from manager.utility import ConvFC 
import torchvision
from manager.t2c import *

from inplace_abn import InPlaceABN as IABN


##########:
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + elu)."""
    
    def __init__(self, name, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, Norm='BN'):
        super(ConvLayer, self).__init__()
        self.g_name = name
        if Norm == 'ABN':
            self.conv = [
                conv_iabn_lrelu(name + '/conv', in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, groups=groups)
            ]        
        else:
            self.conv = [
                conv_bn_relu(name + '/conv', in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, groups=groups)
            ]
        self.conv = nn.Sequential(*self.conv)

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = generate_caffe_prototxt(self.conv, caffe_net, layer)
        return layer

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + elu."""
    
    def __init__(self, name, in_channels, out_channels, stride=1, groups=1, Norm='BN'):
        super(Conv1x1, self).__init__()
        self.g_name = name
        if Norm == 'ABN':
            self.conv = [
                conv_iabn_lrelu(name + '/1x1', in_channels, out_channels, 1, stride=stride, padding=0, groups=groups)
            ]
        else:
            self.conv = [
                conv_bn_relu(name + '/1x1', in_channels, out_channels, 1, stride=stride, padding=0, groups=groups)
            ]
        self.conv = nn.Sequential(*self.conv)

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = generate_caffe_prototxt(self.conv, caffe_net, layer)
        return layer

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""
    
    def __init__(self, name, in_channels, out_channels, stride=1, Norm='BN'):
        super(Conv1x1Linear, self).__init__()
        self.g_name = name
        if Norm == 'ABN':
            self.conv = [
                conv_iabn(name + '/1x1', in_channels, out_channels, 1, stride=stride, padding=0)
            ]
        else:
            self.conv = [
                conv_bn(name + '/1x1', in_channels, out_channels, 1, stride=stride, padding=0)
            ]
        self.conv = nn.Sequential(*self.conv)

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = generate_caffe_prototxt(self.conv, caffe_net, layer)
        return layer

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv3x3(nn.Module):
    """3x3 convolution + bn + elu."""
    
    def __init__(self, name, in_channels, out_channels, stride=1, groups=1, Norm='BN'):
        super(Conv3x3, self).__init__()
        self.g_name = name
        if Norm == 'ABN':
            self.conv = [
                conv_iabn_lrelu(name + '/3x3', in_channels, out_channels, 3, stride=stride, padding=1, groups=groups)
            ]
        else:
            self.conv = [
                conv_bn_relu(name + '/3x3', in_channels, out_channels, 3, stride=stride, padding=1, groups=groups)
            ]
        self.conv = nn.Sequential(*self.conv)

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = generate_caffe_prototxt(self.conv, caffe_net, layer)
        return layer

    def forward(self, x):
        x = self.conv(x)
        return x

class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """
    
    def __init__(self, name, in_channels, out_channels, Norm='BN'):
        super(LightConv3x3, self).__init__()
        self.g_name = name
        self.conv = [
            g_name(name + '/pt', nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)),
            g_name(name + '/dw', nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)),
        ]
        if Norm == 'ABN':
            self.conv.append(g_name(name + '/bn', IABN(out_channels)))
        else:
            self.conv.append(g_name(name + '/bn', nn.BatchNorm2d(out_channels)))

        self.conv = nn.Sequential(*self.conv)

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = generate_caffe_prototxt(self.conv, caffe_net, layer)
        return layer

    def forward(self, x):
        x = self.conv(x)
        return x  


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, name, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        self.g_name = name
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.gate = nn.Sequential(
            g_name(self.g_name + '/gap', nn.AdaptiveAvgPool2d(1)),
            g_name(self.g_name + '/fc1', nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)),
            g_name(self.g_name + '/leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            g_name(self.g_name + '/fc2', nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)),
            g_name(self.g_name + '/sigmoid', nn.Sigmoid())
        )    

    def forward(self, x):
        input = x
        x = self.gate(x)        
        if self.return_gates:
            return x
        return input * x

    def generate_caffe_prototxt(self, caffe_net, layer):
        residual_layer = layer
        layer = generate_caffe_prototxt(self.gate, caffe_net, layer)
        if not self.return_gates:
            layer = L.Flatten(layer, axis=1)
            caffe_net[self.g_name + '/flatten'] = layer
            scale_param = dict(axis=0, bias_term=False)
            layer = L.Scale(residual_layer, layer, scale_param=scale_param)
            caffe_net[self.g_name + '/mul'] = layer
        return layer


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""
    
    def __init__(self, name, in_channels, out_channels, bottleneck_reduction=4, Norm='BN'):
        super(OSBlock, self).__init__()
        self.g_name = name
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(name + '/conv1', in_channels, mid_channels, Norm=Norm)
        self.conv2a = LightConv3x3(name + '/conv2a/1', mid_channels, mid_channels, Norm=Norm)
        self.conv2b = nn.Sequential(
            LightConv3x3(name + '/conv2b/1', mid_channels, mid_channels, Norm=Norm),
            LightConv3x3(name + '/conv2b/2', mid_channels, mid_channels, Norm=Norm),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(name + '/conv2c/1', mid_channels, mid_channels, Norm=Norm),
            LightConv3x3(name + '/conv2c/2', mid_channels, mid_channels, Norm=Norm),
            LightConv3x3(name + '/conv2c/3', mid_channels, mid_channels, Norm=Norm),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(name + '/conv2d/1', mid_channels, mid_channels, Norm=Norm),
            LightConv3x3(name + '/conv2d/2', mid_channels, mid_channels, Norm=Norm),
            LightConv3x3(name + '/conv2d/3', mid_channels, mid_channels, Norm=Norm),
            LightConv3x3(name + '/conv2d/4', mid_channels, mid_channels, Norm=Norm),
        )
        self.gate = ChannelGate(name + '/2a/gate', mid_channels)
        self.conv3 = Conv1x1Linear(name + '/conv3', mid_channels, out_channels, Norm=Norm)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(name + '/downsample', in_channels, out_channels, Norm=Norm)

        if Norm == 'ABN':
            self.activation = nn.Sequential(
                g_name(name + '/leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
            )
        else:
            self.activation = nn.Sequential(
                g_name(name + '/relu', nn.ReLU(inplace=True)),
            )

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = x3 + residual
        
        return self.activation(out)
    
    def generate_caffe_prototxt(self, caffe_net, layer):
        gate1 = self._copy_gate(self.gate, '2b')
        gate2 = self._copy_gate(self.gate, '2c')
        gate3 = self._copy_gate(self.gate, '2d')

        residual_layer = layer
        layer_x1 = generate_caffe_prototxt(self.conv1, caffe_net, layer)

        layer_x2a = generate_caffe_prototxt(self.conv2a, caffe_net, layer_x1)
        layer_x2b = generate_caffe_prototxt(self.conv2b, caffe_net, layer_x1)
        layer_x2c = generate_caffe_prototxt(self.conv2c, caffe_net, layer_x1)
        layer_x2d = generate_caffe_prototxt(self.conv2d, caffe_net, layer_x1)

        layer_ga = generate_caffe_prototxt(self.gate, caffe_net, layer_x2a)
        layer_gb = generate_caffe_prototxt(gate1, caffe_net, layer_x2b)
        layer_gc = generate_caffe_prototxt(gate2, caffe_net, layer_x2c)
        layer_gd = generate_caffe_prototxt(gate3, caffe_net, layer_x2d)

        layer_gate_add = L.Eltwise(layer_ga, layer_gb, layer_gc, layer_gd, operation=P.Eltwise.SUM)
        caffe_net[self.g_name + '/add/gate'] = layer_gate_add

        layer_gate_add = generate_caffe_prototxt(self.conv3, caffe_net, layer_gate_add)
        # layer_x2b = L.Eltwise(layer_x2a, layer_gc, operation=P.Eltwise.SUM)
        # caffe_net[self.g_name + '/add/b'] = layer_x2b
        # layer_x2 = L.Eltwise(layer_x2b, layer_gd, operation=P.Eltwise.SUM)
        # caffe_net[self.g_name + '/add/c'] = layer_x2
        
        if self.downsample is not None:
            residual_layer = generate_caffe_prototxt(self.downsample, caffe_net, layer)

        layer = L.Eltwise(residual_layer, layer_gate_add, operation=P.Eltwise.SUM)
        caffe_net[self.g_name + '/add'] = layer

        layer = generate_caffe_prototxt(self.activation, caffe_net, layer)
        return layer

    def _copy_gate(self, target, name):
        clone = copy.deepcopy(target)
        for _, m in clone.named_modules():
            if hasattr(m, 'g_name'):
                substr = m.g_name.split('/2a/')
                m.g_name = '{}/{}/{}'.format(substr[0], name, substr[1])
        return clone

##########
# Output branch
##########

class Regressor(nn.Module):
    def __init__(self, name, in_channels, out_channels, Norm='BN'):
        super(Regressor, self).__init__()
        self.g_name = name
        self.regressor = [
            g_name(name + '/conv1', nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=1)),
        ]
        if Norm == 'ABN':
            self.regressor.append(g_name(name + '/leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)))
        else:
            self.regressor.append(g_name(name + '/relu', nn.ReLU(inplace=True)))
        
        self.regressor.append(g_name(name + '/conv2', nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=False, groups=1)))

        self.regressor = nn.Sequential(*self.regressor)
    
    def forward(self, x):
        x = self.regressor(x)
        return x

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = generate_caffe_prototxt(self.regressor, caffe_net, layer)
        return layer

##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ArXiv preprint, 2019.
          https://arxiv.org/abs/1905.00953
    """

    def __init__(self, name, num_classes, blocks, layers, channels, feature_dim=512, task='object', num_keypoints=0, Norm='BN'):
        super(OSNet, self).__init__()
        self.g_name = name
        self.task = task
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1 

        # convolutional backbone
        self.conv = nn.Sequential(
            ConvLayer(name + '/block0', 3, channels[0], 7, stride=2, padding=3, Norm=Norm),
            g_name(name + '/block0/reduce/max', nn.MaxPool2d(3, stride=2, padding=1)),            
        )

        self.stage1 = self._make_layer(name + '/block1', blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, Norm=Norm)
        self.stage2 = self._make_layer(name + '/block2', blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True, Norm=Norm)
        self.stage3 = self._make_layer(name + '/block3', blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False, Norm=Norm)

        self.fpn1 = Conv1x1(self.g_name + '/fpn1', channels[1], 256, Norm=Norm)
        self.fpn2 = Conv1x1(self.g_name + '/fpn2', channels[2], 256, Norm=Norm)
        self.fpn3 = Conv1x1(self.g_name + '/fpn3', channels[3], 256, Norm=Norm)
        self.fpn3_upsample = upsample(self.g_name + '/fpn3/upsample' , scale_factor=2, in_channels=256)
        self.fpn2_upsample = upsample(self.g_name + '/fpn2/upsample' , scale_factor=2, in_channels=256)

        self.seg1 = Conv3x3(self.g_name + '/seg1', 256, 128, Norm=Norm)
        self.seg1_upsample   = upsample(self.g_name + '/seg1/upsample'   , scale_factor=2, in_channels=128)
        self.seg2_1 = Conv3x3(self.g_name + '/seg2_1', 256, 128, Norm=Norm)
        self.seg2_2 = Conv3x3(self.g_name + '/seg2_2', 128, 128, Norm=Norm)
        self.seg2_1_upsample = upsample(self.g_name + '/seg2_1/upsample'   , scale_factor=2, in_channels=128)
        self.seg2_2_upsample = upsample(self.g_name + '/seg2_2/upsample'   , scale_factor=2, in_channels=128)
        self.seg3_1 = Conv3x3(self.g_name + '/seg3_1', 256, 128, Norm=Norm)
        self.seg3_2 = Conv3x3(self.g_name + '/seg3_2', 128, 128, Norm=Norm)
        self.seg3_1_upsample = upsample(self.g_name + '/seg3_1/upsample' , scale_factor=2, in_channels=128)
        self.seg3_2_upsample = upsample(self.g_name + '/seg3_2/upsample' , scale_factor=2, in_channels=128)    

        self.hm         = Regressor(self.g_name + '/object_heatmap', 128, num_classes, Norm=Norm)
        self.offset_reg = Regressor(self.g_name + '/object_offset', 128, 2, Norm=Norm)
        self.size_reg   = Regressor(self.g_name + '/object_size', 128, 2, Norm=Norm)

        if task == 'keypoint':
            assert num_keypoints != 0
            self.kp_hm         = Regressor(self.g_name + '/keypoint_heatmap', 128, num_keypoints, Norm=Norm)
            self.kp_offset_reg = Regressor(self.g_name + '/keypoint_offset', 128, 2, Norm=Norm)
            self.kp_loc_reg    = Regressor(self.g_name + '/keypoint_location', 128, num_keypoints * 2, Norm=Norm)

        self._init_params()

    def _make_layer(self, name, block, layer, in_channels, out_channels, reduce_spatial_size, Norm='BN'):
        layers = []
        
        layers.append(block(name + '/0', in_channels, out_channels, Norm=Norm))
        for i in range(1, layer):
            layers.append(block(name + '/{}'.format(i), out_channels, out_channels, Norm=Norm))
        
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(name + '/reduce', out_channels, out_channels, Norm=Norm),
                    g_name(name + '/reduce/avg', nn.AvgPool2d(2, stride=2))
                )
            )
        
        return nn.Sequential(*layers)

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

    def forward(self, x):
        x = self.conv(x)
        # output : C x H/4 x W/4
        stage1 = self.stage1(x)
        # output : C x H/8 x W/8
        stage2 = self.stage2(stage1)
        # output : C x H/16 x W/16
        stage3 = self.stage3(stage2)   

        fpn3 = self.fpn3(stage3)
        fpn2 = self.fpn2(stage2) + fpn3 # self.fpn3_upsample(fpn3)
        fpn1 = self.fpn1(stage1) + self.fpn2_upsample(fpn2)

        seg3_1 = self.seg3_1_upsample(self.seg3_1(fpn3))
        seg3_2 = self.seg3_2_upsample(self.seg3_2(seg3_1))
        seg2_1 = self.seg2_1_upsample(self.seg2_1(fpn2))
        seg2_2 = self.seg2_2_upsample(self.seg2_2(seg2_1)) + seg3_2
        seg1   = self.seg1_upsample(self.seg1(fpn1)) + seg2_2

        ob_hm     = self.hm(seg1)
        ob_offset = self.offset_reg(seg1)
        ob_size   = self.size_reg(seg1)

        if self.task == 'keypoint':
            kp_hm = self.kp_hm(seg1)
            kp_loc = self.kp_loc_reg(seg1)
            kp_offset = self.kp_offset_reg(seg1)
            return ob_hm, ob_offset, ob_size, kp_hm, kp_offset, kp_loc
        else:
            return ob_hm, ob_offset, ob_size

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = generate_caffe_prototxt(self.conv, caffe_net, layer)
        stage1 = generate_caffe_prototxt(self.stage1, caffe_net, layer)
        stage2 = generate_caffe_prototxt(self.stage2, caffe_net, stage1)
        stage3 = generate_caffe_prototxt(self.stage3, caffe_net, stage2)

        stage1 = generate_caffe_prototxt(self.fpn1, caffe_net, stage1)        
        stage2 = generate_caffe_prototxt(self.fpn2, caffe_net, stage2)
        fpn3 = generate_caffe_prototxt(self.fpn3, caffe_net, stage3)

        fpn3_upsample = generate_caffe_prototxt(self.fpn3_upsample, caffe_net, fpn3)
        fpn2 = L.Eltwise(stage2, fpn3_upsample, operation=P.Eltwise.SUM)        
        caffe_net[self.g_name + '/fpn2/add'] = fpn2
        fpn2_upsample = generate_caffe_prototxt(self.fpn2_upsample, caffe_net, fpn2)
        fpn1 = L.Eltwise(stage1, fpn2_upsample, operation=P.Eltwise.SUM)
        caffe_net[self.g_name + '/fpn1/add'] = fpn1

        fpn3 = generate_caffe_prototxt(self.seg3_1, caffe_net, fpn3)
        seg3_1 = generate_caffe_prototxt(self.seg3_1_upsample, caffe_net, fpn3)
        seg3_1 = generate_caffe_prototxt(self.seg3_2, caffe_net, seg3_1)
        seg3_2 = generate_caffe_prototxt(self.seg3_2_upsample, caffe_net, seg3_1)
        fpn2 = generate_caffe_prototxt(self.seg2_1, caffe_net, fpn2)
        seg2_1 = generate_caffe_prototxt(self.seg2_1_upsample, caffe_net, fpn2)
        seg2_1 = generate_caffe_prototxt(self.seg2_2, caffe_net, seg2_1)
        seg2_2 = generate_caffe_prototxt(self.seg2_2_upsample, caffe_net, seg2_1)
        seg2_2 = L.Eltwise(seg2_2, seg3_2, operation=P.Eltwise.SUM)        
        caffe_net[self.g_name + '/seg2/add'] = seg2_2
        fpn1 = generate_caffe_prototxt(self.seg1, caffe_net, fpn1)
        seg1 = generate_caffe_prototxt(self.seg1_upsample, caffe_net, fpn1)
        seg1 = L.Eltwise(seg2_2, seg1, operation=P.Eltwise.SUM)
        caffe_net[self.g_name + '/seg1/add'] = seg1
        
        ob_hm = generate_caffe_prototxt(self.hm_reg, caffe_net, seg1)
        ob_offset = generate_caffe_prototxt(self.offset_reg, caffe_net, seg1)
        ob_size = generate_caffe_prototxt(self.size_reg, caffe_net, seg1)

        if self.task == 'keypoint':
            kp_hm = generate_caffe_prototxt(self.kp_hm_reg, caffe_net, seg1)
            kp_loc = generate_caffe_prototxt(self.kp_loc_reg, caffe_net, seg1)
            kp_offset = generate_caffe_prototxt(self.kp_offset_reg, caffe_net, seg1)
            return ob_hm, ob_offset, ob_size, kp_hm, kp_offset, kp_loc
        else:
            return ob_hm, ob_offset, ob_size
        

##########
# Instantiation
##########
def get_osnet_pfpn(cfg):
    model = OSNet(cfg.MODEL.NAME, 
                  cfg.DB.NUM_CLASSES, 
                  blocks=[OSBlock, OSBlock, OSBlock], 
                  layers=[2, 2, 2],
                  channels=[64, 256, 384, 512], 
                  feature_dim=512, 
                  task=cfg.TASK, 
                  num_keypoints=cfg.DB.NUM_KEYPOINTS*2, 
                  Norm=cfg.MODEL.NORM)
    return model
