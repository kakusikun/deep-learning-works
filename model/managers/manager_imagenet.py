import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.OSNetv2 import osnet_x1_0
from model.RMNet import RMNet
from model.ResNet import ResNet, BasicBlock
from model.utility import ConvFC, CrossEntropyLossLS
from model.model_manager import TrainingManager
import logging
logger = logging.getLogger("logger")

class ImageNetManager(TrainingManager):
    def __init__(self, cfg):
        super(ImageNetManager, self).__init__(cfg)        

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

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
            
class Model(nn.Module):
    def __init__(self, num_classes, model_name):
        super(Model, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.backbone = ResNet(last_stride=1, block=BasicBlock, layers=[2,2,2,2])
        elif model_name == 'rmnet':
            self.in_planes = 256
            self.backbone = RMNet(b=[4,8,10,11], cifar10=False, reid=True, trick=True)
        elif model_name == 'osnet':
            self.in_planes = 512
            self.backbone = osnet_x1_0(num_classes, loss='trick')
        else:
            logger.info("{} is not supported".format(model_name))

        self.num_classes = num_classes
        self.fc = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.fc.apply(weights_init_classifier)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)      
        global_feat = self.fc(x)
        return global_feat
