import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.OSNetv2 import osnet_x1_0, osnet_att_x1_0
from model.RMNet import RMNet
from model.ResNet import ResNet, BasicBlock
from model.utility import ConvFC, CenterLoss, AMSoftmax, CrossEntropyLossLS, TripletLoss, AttentionConvBlock
from model.manager import TrainingManager
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
        if self.cfg.MODEL.NAME == 'osnet':
            feat_dim = 512 + 1024
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))
            sys.exit(1)

        ce_ls = CrossEntropyLossLS(self.cfg.MODEL.NUM_CLASSES)
        center_loss = CenterLoss(feat_dim, self.cfg.MODEL.NUM_CLASSES, len(self.cfg.MODEL.GPU) > 0 and torch.cuda.is_available())        
        triplet_loss = TripletLoss()
        self.loss_has_param = [center_loss]
        self.loss_name = ["cels", "triplet", "center"]

        def loss_func(l_feat, g_feat, target):
            each_loss = [ce_ls(g_feat, target),
                         triplet_loss(l_feat, target)[0], 
                         center_loss(l_feat, target)]
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
        self.cfg = cfg 
        if cfg.MODEL.NAME == 'osnet':
            self.in_planes = 512
            if cfg.MODEL.PRETRAIN == "outside":
                self.backbone = osnet_att_x1_0(pooling=cfg.MODEL.POOLING, task=cfg.MODEL.TASK) 
            else:
                self.backbone = osnet_att_x1_0(num_classes=cfg.MODEL.NUM_CLASSES, pooling=cfg.MODEL.POOLING, task=cfg.MODEL.TASK)        
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))
            sys.exit(1)

        if cfg.MODEL.POOLING == 'AVG':
            self.gp = nn.AdaptiveAvgPool2d(1)   
        elif cfg.MODEL.POOLING == 'MAX':
            self.gp = nn.AdaptiveMaxPool2d(1)
        else:
            logger.info("{} is not supported".format(cfg.MODEL.POOLING))
            sys.exit(1)

        self.BNNeck = nn.BatchNorm2d(self.in_planes + 1024)
        self.BNNeck.bias.requires_grad_(False)  # no shift
        self.BNNeck.apply(weights_init_kaiming)

        self.att_conv_block = AttentionConvBlock(384, 1024)
        self.att_conv_block.apply(weights_init_kaiming)

        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.id_fc = nn.Linear(self.in_planes + 1024, self.num_classes, bias=False)        
        self.id_fc.apply(weights_init_classifier)
    
    def forward(self, x):
        if self.cfg.MODEL.TASK == 'trick':
            # use trick: BNNeck, feature before BNNeck to triplet, but GMP
            feat, at_map = self.backbone(x)

            # additional attention incorporation
            at_map = self.att_conv_block(at_map)
            at_map = self.gp(at_map)        
            x = self.gp(feat)
            # fused before BNNeck
            x = torch.cat((x, at_map), dim=1)
            # feature to triplet loss
            local_feat = x.view(x.size(0), -1)

            x = self.BNNeck(x)
            x = x.view(x.size(0), -1)
            if not self.training:
                return x        
            global_feat = self.id_fc(x)
            return local_feat, global_feat

        if self.cfg.MODEL.TASK == 'attention':
            # use trick: BNNeck, feature after BNNeck to triplet, but GMP and feature after fc in backbone
            feat, at_map = self.backbone(x)

            # additional attention incorporation
            at_map = self.att_conv_block(at_map)
            at_map = self.gp(at_map)        

            # fuse
            x = torch.cat((feat, at_map), dim=1)
            x = self.BNNeck(x)

            # feature to triplet loss
            local_feat = x.view(x.size(0), -1)
            if not self.training:
                return local_feat        

            global_feat = self.id_fc(local_feat)
            return local_feat, global_feat
