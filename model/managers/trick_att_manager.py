import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.OSNetv2 import osnet_x1_0, osnet_ibn_x1_0, osnet_att_x1_0
from model.RMNet import RMNet
from model.ResNet import ResNet, BasicBlock
from model.utility import ConvFC, CenterLoss, AMSoftmax, CrossEntropyLossLS, TripletLoss, AttentionConvBlock
from model.model_manager import TrainingManager
import logging
logger = logging.getLogger("logger")

class AttentionManager(TrainingManager):
    def __init__(self, cfg):
        super(AttentionManager, self).__init__(cfg)        

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
        if self.cfg.MODEL.NAME == 'resnet18' or self.cfg.MODEL.NAME == 'osnet' or self.cfg.MODEL.NAME == 'osnetibn':
            feat_dim = 512
        elif self.cfg.MODEL.NAME == 'rmnet':
            feat_dim = 256        

        ce_ls = CrossEntropyLossLS(self.cfg.MODEL.NUM_CLASSES)
        center_loss = CenterLoss(feat_dim, self.cfg.MODEL.NUM_CLASSES, self.cfg.MODEL.NUM_GPUS > 0 and torch.cuda.is_available())        
        triplet_loss = TripletLoss()
        attention_loss = nn.BCEWithLogitsLoss()
        self.loss_has_param = [center_loss]
        self.loss_name = ["cels", "triplet", "center", "attention"]

        def loss_func(l_feat, g_feat, at_map, target, at_map_gt, at_map_keys):
            keys = at_map_keys == 1
            each_loss = [ce_ls(g_feat, target),
                         triplet_loss(l_feat, target)[0], 
                         center_loss(l_feat, target), 
                         attention_loss(at_map[keys], at_map_gt[keys])]
            loss = each_loss[0] + each_loss[1] + self.cfg.SOLVER.CENTER_LOSS_WEIGHT * each_loss[2] + each_loss[3]
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
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
            
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        
        if cfg.MODEL.NAME == 'osnet':
            self.in_planes = 512
            if cfg.MODEL.PRETRAIN == "outside":
                self.backbone = osnet_att_x1_0(loss='trick') 
            else:
                self.backbone = osnet_att_x1_0(num_classes=cfg.MODEL.NUM_CLASSES, loss='trick')        
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))

        self.gmp = nn.AdaptiveMaxPool2d(1)        
        self.BNNeck = nn.BatchNorm2d(self.in_planes)
        self.BNNeck.bias.requires_grad_(False)  # no shift
        self.BNNeck.apply(weights_init_kaiming)

        self.att_conv_block = AttentionConvBlock(512, 1024)
        self.att_conv_block.apply(weights_init_kaiming)

        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.id_fc = nn.Linear(self.in_planes + 1024, self.num_classes, bias=False)        
        self.id_fc.apply(weights_init_classifier)
    
    def forward(self, x):
        feat, at_map = self.backbone(x)

        # additional attention incorporation
        at_map = self.att_conv_block(at_map)
        at_map = self.gmp(at_map)        
        x = self.gmp(feat)
        # fused before BNNeck
        x = torch.cat((x, at_map), dim=1)
        # feature to triplet loss
        local_feat = x.view(x.size(0), -1)

        x = self.BNNeck(x)
        x = x.view(x.size(0), -1)
        if not self.training:
            return x        
        global_feat = self.id_fc(x)
        return local_feat, global_feat, at_map
