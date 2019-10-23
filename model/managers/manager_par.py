import os
import sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.OSNetv2 import osnet_x1_0
from model.RMNet import RMNet
from model.ResNet import ResNet, BasicBlock
from model.utility import ConvFC, CenterLoss, AMSoftmax, CrossEntropyLossLS, TripletLoss
from model.manager import TrainingManager
import logging
logger = logging.getLogger("logger")

class PARManager(TrainingManager):
    def __init__(self, cfg):
        super(PARManager, self).__init__(cfg)        
        self.category_names = ['gender', 'hair', 'shirt', 'plaid', 'stripe', 'sleeve',
                               'logo', 'shorts', 'skirt', 'hat', 'glasses', 'backpack', 
                               'bag']
        self.alpha = -4.45
        self.beta = 5.45

        if cfg.TASK == "par":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()       
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        if self.cfg.MODEL.NAME == 'osnet':
            feat_dim = 512
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))
            sys.exit(1)       

        bce = nn.BCEWithLogitsLoss(reduction='none')

        self.loss_name = ["BCE_{}".format(c) for c in range(self.cfg.MODEL.NUM_CLASSES)]

        def loss_func(feat, target):
            temp_target = torch.zeros_like(target)            
            temp_target[target>0] = 1            
            known_target = (target>-1).float()
            p = self.alpha * known_target.mean(dim=1).reshape(known_target.size(0),1) + self.beta    
            loss_table = p * bce(feat, temp_target)
            each_loss = torch.Tensor([loss_table[:,i][known_target[:,i]==1].mean() for i in range(loss_table.shape[1])])
            each_loss[each_loss != each_loss] = -1
            loss = loss_table[known_target==1].mean()
                    
            predicted_target = torch.zeros_like(feat)
            predicted_target[feat.sigmoid()>=0.5] = 1
            accu = (predicted_target[known_target==1] == target[known_target==1]).sum() / known_target.sum()
            return loss, each_loss, accu

        self.loss_func = loss_func

class SinglePARManager(TrainingManager):
    def __init__(self, cfg):
        super(SinglePARManager, self).__init__(cfg)        
        
        if cfg.TASK == "par":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()          
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        if self.cfg.MODEL.NAME == 'osnet':
            feat_dim = 512
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))
            sys.exit(1)       

        bce = nn.BCEWithLogitsLoss()

        self.loss_name = ["BCE"]

        def loss_func(feat, target):
            loss = bce(feat, target)
            each_loss = [loss]
            predicted_target = torch.zeros_like(feat)
            predicted_target[feat.sigmoid()>=0.5] = 1
            accu = (predicted_target == target).float().mean()
            return loss, each_loss, accu

        self.loss_func = loss_func

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
        if cfg.MODEL.NAME == 'osnet':
            self.in_planes = 512
            if cfg.MODEL.PRETRAIN == "outside":
                self.backbone = osnet_x1_0(task=cfg.MODEL.TASK)  
            else:
                self.backbone = osnet_x1_0(cfg.MODEL.NUM_CLASSES, task=cfg.MODEL.TASK)        
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))
     
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.attr_fc = nn.Linear(self.in_planes, self.num_classes)        
        self.attr_fc.apply(weights_init_classifier)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)
        x = self.attr_fc(x)
        if not self.training:
            return x.sigmoid()
        return x
