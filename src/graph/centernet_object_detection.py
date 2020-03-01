from src.graph import *
from tools.utils import _sigmoid

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg)
        self.feature_extraction = biFPN(self.backbone.stage_out_channels[-3:], 24, fpn_tail=True)
        self.wh_head = ReshapeHead(24 * 3, 2)
        self.heat_head = ReshapeHead(24 * 3, self.cfg.DB.NUM_CLASSES)
        self.reg_head = ReshapeHead(24 * 3, 2)      
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
        self.crit = {}
        self.crit['hm'] = FocalLoss()  
        self.crit['wh'] = SmoothL1Loss()
        self.crit['reg'] = SmoothL1Loss()
        self.loss_w = nn.Parameter(torch.Tensor([0.0, 0.0, -1.0 * torch.log(0.2), -1.0 * torch.log(2)]))
    def forward(self, feats, batch):
        hm_loss = 0.0
        wh_loss = 0.0
        reg_loss = 0.0
        for feat in feats:
            for head in feat.keys():
                output = feat[head]
                if head == 'hm':
                    output = _sigmoid(output)
                    hm_loss += self.crit[head](self.loss_w[:2], output, batch['hm'])                 
                elif head == 'wh':
                    wh_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['wh']) * torch.exp(-1.0 * self.loss_w[2]) * 0.5
                elif head == 'reg':
                    reg_loss += self.crit[head](output, batch['reg_mask'], batch['ind'], batch['reg']) * torch.exp(-1.0 * self.loss_w[3]) * 0.5
                else:
                    raise TypeError
        losses = {'hm':hm_loss, 'wh':wh_loss, 'reg':reg_loss}
        loss = losses['hm'] + losses['wh'] + losses['reg'] + self.loss_w.sum()
        return loss, losses

class CenterNetObjectDetection(BaseGraph):
    def __init__(self, cfg):
        super(CenterNetObjectDetection, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)        
        self.loss_head = _LossHead()
            
