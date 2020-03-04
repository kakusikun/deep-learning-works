from __future__ import absolute_import
from __future__ import division

__all__ = ['MobileNetv1']

import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from manager.t2c import *

class MobileNetv1(nn.Module):
    def __init__(self, num_classes=1000, isbackbone=False, alpha=1):
        super(MobileNetv1, self).__init__()

        def conv_bn(name, inp, oup, stride):
            return conv_bn_relu(name=name, in_channels=inp, out_channels=oup, kernel_size=3, stride=stride, padding=1)

        def conv_dw(name, inp, oup, stride):
            conv_dw = [
                conv_bn_relu(name=name+"/dw", in_channels=inp, out_channels=inp, kernel_size=3, stride=stride, groups=inp, padding=1),
                conv_bn_relu(name=name+"/sep", in_channels=inp, out_channels=oup, kernel_size=1, stride=1)
            ]
            return nn.Sequential(*conv_dw)

        setting = [   
            (int(  3), int( 32*alpha), 2), 
            (int( 32*alpha), int( 64*alpha), 1),
            (int( 64*alpha), int(128*alpha), 2),
            (int(128*alpha), int(128*alpha), 1),
            (int(128*alpha), int(256*alpha), 2),
            (int(256*alpha), int(256*alpha), 1),
            (int(256*alpha), int(512*alpha), 2),
            (int(512*alpha), int(512*alpha), 1),
            (int(512*alpha), int(512*alpha), 1),
            (int(512*alpha), int(512*alpha), 1),
            (int(512*alpha), int(512*alpha), 1),
            (int(512*alpha), int(512*alpha), 1),
            (int(512*alpha), int(1024*alpha), 2),
            (int(1024*alpha), int(1024*alpha), 1)         
        ]

        model = []
        for i, config in enumerate(setting):
            inp, oup, s = config
            if i == 0:
                model.append(conv_bn("conv{}".format(i), inp, oup, s))
            else:
                model.append(conv_dw("conv{}".format(i), inp, oup, s))
        if not isbackbone:
            model.append(g_name('pool', nn.AdaptiveAvgPool2d(1)))
            model.append(flatten('flatten', 1))
            model.append(g_name('fc', nn.Linear(int(1024*alpha), num_classes)))

        self.model = nn.Sequential(*model)
        
    def generate_caffe_prototxt(self, caffe_net, layer):
        return generate_caffe_prototxt(self.model, caffe_net, layer)

    def convert_to_caffe(self, name, path, input_size):
        def assert_diff(a, b):
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            if isinstance(b, torch.Tensor):
                b = b.detach().cpu().numpy()
            print(a.shape, b.shape)
            a = a.reshape(-1)
            b = b.reshape(-1)
            assert a.shape == b.shape
            print(a, b)
            diff = np.abs(a - b)
            print('mean diff = %f' % diff.mean())
            assert diff.mean() < 0.001
            print('max diff = %f' % diff.max())
            assert diff.max() < 0.001

        caffe_net = caffe.NetSpec()
        layer = L.Input(shape=dict(dim=input_size))
        caffe_net.tops['data'] = layer
        generate_caffe_prototxt(self, caffe_net, layer)
        print(caffe_net.to_proto())
        with open(osp.join(path, "{}.prototxt".format(name)), 'wb') as f:
            f.write(str(caffe_net.to_proto()))
        caffe_net = caffe.Net(osp.join(path, "{}.prototxt".format(name)), caffe.TEST)
        convert_pytorch_to_caffe(self, caffe_net)
        caffe_net.save(osp.join(path, "{}.caffemodel".format(name)))
        self.caffe_net = caffe_net
        img = np.random.rand(*input_size)
        x = torch.tensor(img.copy(), dtype=torch.float32)
        
        self.train(False)
        with torch.no_grad():
            cls_results = self.model(x)

        caffe_net.blobs['data'].data[...] = img.copy()
        caffe_results = caffe_net.forward()
        blob_name = list(caffe_results.keys())[0]
        cls_results_caffe = caffe_results[blob_name]
        assert_diff(cls_results, cls_results_caffe)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    net = MobileNetv1()
    net.convert_to_caffe('MobileNetV1')