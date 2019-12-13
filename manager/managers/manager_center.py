import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.model_factory import get_model
from manager.utility import FocalLoss, RegL1Loss
from manager.base_manager import BaseManager
from tools.utils import _sigmoid
import logging
logger = logging.getLogger("logger")

class CenterManager(BaseManager):
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

        def loss_func(feats, batch):
            focal_loss = 0.0
            wh_loss    = 0.0
            off_loss   = 0.0
            
            for feat in feats:
                for head in feat.keys():
                    if head == 'hm':
                        output = _sigmoid(feat[head])
                        focal_loss += focal(output, batch['hm'])
                    elif head == 'wh':
                        output = feat[head]
                        wh_loss += regli(output, batch['reg_mask'], batch['ind'], batch['wh'])
                    else:
                        output = feat[head]
                        off_loss += regli(output, batch['reg_mask'], batch['ind'], batch['reg'])

            each_loss = [focal_loss, wh_loss, off_loss]
            loss = each_loss[0] + 0.1 * each_loss[1] + each_loss[2]
            return loss, each_loss

        self.loss_func = loss_func
            
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = get_model(cfg.MODEL.NAME)(cfg)
    
    def forward(self, x):
        out = self.backbone(x)
        return out
