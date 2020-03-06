from src.graph import *
import math
from tools.utils import _sigmoid

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg)
        self.feature_extraction = biFPN(self.backbone.stage_out_channels[-3:], 24, fpn_tail=True)
        self.wh_head = RegressionHead(24 * 3, 2)
        self.heat_head = RegressionHead(24 * 3, cfg.DB.NUM_CLASSES)
        self.reg_head = RegressionHead(24 * 3, 2)      
    def forward(self, x):
        ps = self.backbone(x)
        feat = self.feature_extraction(ps[-3:])
        outputs = {
            'hm': self.heat_head(feat),
            'wh': self.wh_head(feat),
            'reg': self.reg_head(feat)
        }
        return outputs

class _LossHead(nn.Module):
    def __init__(self):
        super(_LossHead, self).__init__()
        self.crit = {}
        self.crit['hm'] = FocalLoss()  
        self.crit['wh'] = SmoothL1Loss()
        self.crit['reg'] = SmoothL1Loss()
    def forward(self, feats, batch):
        hm_loss = 0.0
        wh_loss = 0.0
        reg_loss = 0.0
        for head in feats:
            output = feats[head]
            if head == 'hm':
                output = _sigmoid(output)
                hm_loss += self.crit[head](output, batch['hm'])                 
            elif head == 'wh':
                wh_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['wh']) * 0.1
            elif head == 'reg':
                reg_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['reg'])
            else:
                raise TypeError
        losses = {'hm':hm_loss, 'wh':wh_loss, 'reg':reg_loss, }
        loss = losses['hm'] + losses['wh'] + losses['reg']
        return loss, losses

class CenterNetObjectDetection(BaseGraph):
    def __init__(self, cfg):
        super(CenterNetObjectDetection, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)     
        self.loss_head = _LossHead()
        self.sub_models['loss'] = self.loss_head
            
