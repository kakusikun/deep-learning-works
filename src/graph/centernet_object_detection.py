from src.graph import *
import math
from tools.utils import _sigmoid

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg)
        self.feature_extraction = biFPN(["i_0", "i_1", "i_2"], self.backbone.stage_out_channels[-3:], [1,2,4], 32, fpn=True, weighted_resize=True)
        self.wh_head = RegressionHead(32 * len(self.feature_extraction.configs), 2)
        self.heat_head = RegressionHead(32 * len(self.feature_extraction.configs), cfg.DB.NUM_CLASSES)
        self.reg_head = RegressionHead(32 * len(self.feature_extraction.configs), 2)      
    def forward(self, x):
        ps = self.backbone(x)
        feat = self.feature_extraction(ps[-1*len(self.feature_extraction.configs):])
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
        self.crit['wh'] = L1Loss()
        self.crit['reg'] = L1Loss()
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
                wh_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['wh'])
            elif head == 'reg':
                reg_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['reg'])
            else:
                raise TypeError
        losses = {'hm':hm_loss, 'wh':wh_loss, 'reg':reg_loss, }
        loss = losses['hm'] + losses['wh'] * 0.1 + losses['reg']
        return loss, losses

class CenterNetObjectDetection(BaseGraph):
    def __init__(self, cfg):
        super(CenterNetObjectDetection, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)     
        self.loss_head = _LossHead()
