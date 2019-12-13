import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.model_factory import get_model
from manager.utility import ConvFC, CenterLoss, AMSoftmax, CrossEntropyLossLS, TripletLoss
from manager.base_manager import BaseManager
import logging
logger = logging.getLogger("logger")

class TrickManager(BaseManager):
    def __init__(self, cfg):
        super(TrickManager, self).__init__(cfg)        

        if cfg.TASK == "reid":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()           
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        ce_ls = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)
        center_loss = CenterLoss(512, self.cfg.DB.NUM_CLASSES)        
        triplet_loss = TripletLoss()
        self.loss_has_param = [center_loss]
        self.loss_name = ["cels", "triplet", "center"]

        def loss_func(l_feat, g_feat, target):
            each_loss = [ce_ls(g_feat, target), triplet_loss(l_feat, target)[0], center_loss(l_feat, target)]
            loss = each_loss[0] + each_loss[1] + self.cfg.SOLVER.CENTER_LOSS_WEIGHT * each_loss[2]
            return loss, each_loss

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
        self.in_planes = 512
        if cfg.MODEL.PRETRAIN == "outside":
            self.backbone = get_model(cfg.MODEL.NAME)(task='trick') 
        else:
            self.backbone = get_model(cfg.MODEL.NAME)(cfg.DB.NUM_CLASSES, task='trick') 

        self.gap = nn.AdaptiveAvgPool2d(1)        
        self.BNNeck = nn.BatchNorm1d(self.in_planes)
        self.BNNeck.bias.requires_grad_(False)  # no shift
        self.BNNeck.apply(weights_init_kaiming)

        self.num_classes = cfg.DB.NUM_CLASSES
        self.id_fc = nn.Linear(self.in_planes, self.num_classes, bias=False)        
        self.id_fc.apply(weights_init_classifier)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        feat = self.backbone(x)
        x = self.gap(feat)
        local_feat = x.view(x.size(0), -1)
        x = self.BNNeck(local_feat)
        #  x = x.view(x.size(0), -1)        
        if not self.training:
            return x      
        global_feat = self.id_fc(x)  
        return local_feat, global_feat
