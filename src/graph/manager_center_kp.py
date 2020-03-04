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

                    
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        self.crit = {}
        self.crit['hm'] = FocalLoss()  
        self.crit['wh'] = RegL1Loss()
        self.crit['reg'] = RegL1Loss()
        self.crit['hm_kp'] = FocalLoss()  
        self.crit['kps'] = RegWeightedL1Loss()
        self.crit['kp_reg'] = RegL1Loss()        

        def loss_func(feats, batch):
            hm_loss    = 0.0
            wh_loss    = 0.0
            off_loss   = 0.0
            hm_kp_loss = 0.0
            hp_loss    = 0.0
            hp_off_loss = 0.0
            
            for feat in feats:
                for head in feat.keys():
                    output = feat[head]
                    if head == 'hm':
                        output = _sigmoid(output)
                        hm_loss += self.crit[head](output, batch['hm'])                 
                    elif head == 'wh':
                        wh_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['wh'])
                    elif head == 'reg':
                        off_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['reg'])
                    elif head == 'hm_kp':
                        output = _sigmoid(output)
                        hm_kp_loss += self.crit[head](output, batch['hm_kp'])  
                    elif head == 'kp_reg':
                        hp_off_loss += self.crit[head](output, batch['kp_mask'], batch['kp_ind'], batch['kp_reg'])
                    elif head == 'kps':
                        hp_loss += self.crit[head](output, batch['kps_mask'], batch['ind'], batch['kps'])
                    else:
                        sys.exit(1)
            each_loss = {'hm':hm_loss, 'wh':wh_loss, 'reg':off_loss, 'hm_kp':hm_kp_loss, 'kps':hp_loss, 'kp_reg':hp_off_loss}
            loss = each_loss['hm'] + 0.1 * each_loss['wh'] + each_loss['reg'] + each_loss['hm_kp'] + each_loss['kps'] + each_loss['kp_reg']
            return loss, each_loss

        self.loss_func = loss_func
            
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.backbone = get_model(cfg.MODEL.NAME)(cfg)  
            
    def forward(self, x):
        out = self.backbone(x)
        return out
