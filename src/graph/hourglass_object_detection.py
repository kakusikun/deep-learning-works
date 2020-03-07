from src.graph import *
import math
from tools.utils import _sigmoid

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg)
        self.heads = {
            'hm': nn.ModuleList([HourGlassHead(256, cfg.DB.NUM_CLASSES), HourGlassHead(256, cfg.DB.NUM_CLASSES)]),
            'wh': nn.ModuleList([HourGlassHead(256, 2), HourGlassHead(256, 2)]),
            'reg': nn.ModuleList([HourGlassHead(256, 2), HourGlassHead(256, 2)])
        }
        for m in self.heads['hm']:
            m[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        outs = self.backbone(x)
        outputs = []
        for i, out in enumerate(outs):
            head_out = {}
            for head in self.heads:
                head_out[head] = self.heads[head][i](out)
            outputs.append(head_out)
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
        for feat in feats:
            for head in feat:
                output = feat[head]
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
            
