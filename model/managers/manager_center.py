import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.OSNet_PFPN import osnet_x1_0
from model.utility import FocalLoss, RegL1Loss
from model.manager import TrainingManager
from tools.utils import _sigmoid
import logging
logger = logging.getLogger("logger")

class CenterManager(TrainingManager):
    def __init__(self, cfg):
        super(CenterManager, self).__init__(cfg)        

        if cfg.TASK == "object":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()           
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        regli = RegL1Loss()
        focal = FocalLoss()    
        self.loss_has_param = []
        self.loss_name = ["focal", "reg_wh", "reg_offset"]

        def loss_func(feats, targets):
            ob_hm, ob_offset, ob_size = feats
            hm, wh, reg, reg_mask, ind = targets

            focal_loss = focal(ob_hm, hm)
            wh_loss    = regli(ob_size  , reg_mask, ind, wh)
            off_loss   = regli(ob_offset, reg_mask, ind, reg)

            each_loss = [focal_loss, wh_loss, off_loss]
            loss = each_loss[0] + 0.1 * each_loss[1] + each_loss[2]
            return loss, each_loss

        self.loss_func = loss_func
            
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        if cfg.MODEL.NAME == 'osnet-center':
            self.backbone = osnet_x1_0(cfg.MODEL.NAME, cfg.MODEL.NUM_CLASSES, task='object')        
        else:
            logger.info("{} is not supported".format(cfg.MODEL.NAME))
    
    def forward(self, x):
        ob_hm, ob_offset, ob_size = self.backbone(x)
        return _sigmoid(ob_hm), ob_offset, ob_size
