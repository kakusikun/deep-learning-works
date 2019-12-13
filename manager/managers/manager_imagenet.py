import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.model_factory import get_model
from manager.utility import ConvFC, CrossEntropyLossLS
from manager.base_manager import BaseManager
import logging
logger = logging.getLogger("logger")

class ImageNetManager(BaseManager):
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
        self.model = Model(self.cfg)

    def _make_loss(self):
        #  ce_ls = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)
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
            
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = get_model(cfg.MODEL.NAME)(cfg.DB.NUM_CLASSES, task='classifier')

        self.backbone.classifier.apply(weights_init_classifier)
    
    def forward(self, x):
        x = self.backbone(x)
        return x
