import os
import sys
import torch
import math
import torch.nn as nn
from collections import OrderedDict
from model.model_factory import get_model
from manager.utility import FocalLoss, RegL1Loss, RegWeightedL1Loss
from manager.base_manager import BaseManager
from tools.utils import _sigmoid
import logging
logger = logging.getLogger("logger")

class CenterKPManager(BaseManager):
    def __init__(self, cfg):
        super(CenterKPManager, self).__init__(cfg)        

        if cfg.TASK == "keypoint":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

        self._check_model()           
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        regl1 = RegL1Loss()
        regwl1 = RegWeightedL1Loss()
        focal = FocalLoss()    
        self.loss_has_param = []
        self.loss_name = ["hm", "wh", "offset", "hm_hp", "hp", "hp_offset"]

        def loss_func(feats, batch):
            hm_loss    = 0.0
            wh_loss    = 0.0
            off_loss   = 0.0
            hm_hp_loss = 0.0
            hp_loss    = 0.0
            hp_off_loss = 0.0
            
            for feat in feats:
                for head in feat.keys():
                    output = feat[head]
                    if head == 'hm':
                        output = _sigmoid(output)
                        hm_loss += focal(output, batch['hm'])                 
                    elif head == 'wh':
                        wh_loss += regl1(output, batch['reg_mask'], batch['ind'], batch['wh'])
                    elif head == 'reg':
                        off_loss += regl1(output, batch['reg_mask'], batch['ind'], batch['reg'])
                    elif head == 'hm_hp':
                        output = _sigmoid(output)
                        hm_hp_loss += focal(output, batch['hm_hp'])  
                    elif head == 'hp_offset':
                        hp_off_loss += regl1(output, batch['hp_mask'], batch['hp_ind'], batch['hp_reg'])
                    elif head == 'hps':
                        hp_loss += regwl1(output, batch['hps_mask'], batch['ind'], batch['hps'])
                    else:
                        sys.exit(1)
            each_loss = [hm_loss, wh_loss, off_loss, hm_hp_loss, hp_loss, hp_off_loss]
            loss = each_loss[0] + 0.1 * each_loss[1] + each_loss[2] + each_loss[3] + each_loss[4] + each_loss[5]
            return loss, each_loss

        self.loss_func = loss_func
            
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = get_model(cfg.MODEL.NAME)(cfg)  
            
    def forward(self, x):
        out = self.backbone(x)
        return out
