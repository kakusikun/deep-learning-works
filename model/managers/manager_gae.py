import os
import os.path as osp
import sys
import numpy as np
import math
from collections import OrderedDict

import torch
import torch.nn as nn

from model.MobileNetv1 import MobileNetv1
from model.t2c import *
from model.model_manager import TrainingManager
import logging
logger = logging.getLogger("logger")

class GAEManager(TrainingManager):
    def __init__(self, cfg):
        super(GAEManager, self).__init__(cfg)        

        if cfg.TASK == "imagenet":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()    
        
        self.loss_name = ["cels"]
                        
    def _make_model(self):
        self.model = Model(self.cfg.MODEL.NUM_CLASSES, self.cfg.MODEL.NAME)

    def _make_loss(self):
        #  ce_ls = CrossEntropyLossLS(self.cfg.MODEL.NUM_CLASSES)
        ce = nn.CrossEntropyLoss()

        self.loss_has_param = []

        def loss_func(g_feat, target):
            #  each_loss = [ce_ls(g_feat, target)]            
            each_loss = [ce(g_feat, target)]            
            loss = each_loss[0]
            return loss, each_loss

        self.loss_func = loss_func

    def _initialize_weights(self):
        pass

def weights_init_kaiming(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

class GAE(nn.Module):
    def __init__(self, num_classes):
        super(GAE, self).__init__()
        in_features = 4*4*1024
        
        self.gender = g_name("gender", nn.Linear(in_features, num_classes[0]))
        self.age = g_name("age", nn.Linear(in_features, num_classes[1]))
        self.emotion = g_name("emotion", nn.Linear(in_features, num_classes[2]))
        self.flatten = flatten("reshape", 1)

    def forward(self, x):
        x = self.flatten(x)
        gender = self.gender(x)
        age = self.age(x)
        emotion = self.emotion(x)
        return gender, age, emotion

    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = self.flatten.generate_caffe_prototxt(caffe_net, layer)
        gender_layer = generate_caffe_prototxt(self.gender, caffe_net, layer)
        age_layer = generate_caffe_prototxt(self.age, caffe_net, layer)
        emotion_layer = generate_caffe_prototxt(self.emotion, caffe_net, layer)

class Model(nn.Module):
    def __init__(self, num_classes, model_name):
        super(Model, self).__init__()
        if model_name != 'gae':
            logger.info("{} is not supported".format(model_name))
            sys.exit(1)

        self.backbone = MobileNetv1(isbackbone=True)
        self.gae = GAE(num_classes)

        self.backbone.apply(weights_init_kaiming)
        self.gae.apply(weights_init_classifier)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.gae(x)
        return x
    
    def generate_caffe_prototxt(self, caffe_net, layer):
        layer = self.backbone.generate_caffe_prototxt(caffe_net, layer)
        layer = self.gae.generate_caffe_prototxt(caffe_net, layer)
        
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
            x = self.backbone(x)
            gender, age, emotion = self.gae(x)

        caffe_net.blobs['data'].data[...] = img.copy()
        caffe_results = caffe_net.forward()
        gender_caffe, age_caffe, emotion_caffe = caffe_results['gender'], caffe_results['age'], caffe_results['emotion']

        assert_diff(gender, gender_caffe)
        assert_diff(age, age_caffe)
        assert_diff(emotion, emotion_caffe)

if __name__ == '__main__':
    net = Model(num_classes=[1,101,3], model_name='gae')
    net.convert_to_caffe('GAE')