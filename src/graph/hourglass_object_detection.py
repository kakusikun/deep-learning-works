from src.graph import *
import math
from tools.utils import _sigmoid

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        w, h = cfg.INPUT.SIZE
        self.out_sizes = [(w // s, h // s) for s in cfg.MODEL.STRIDES]
        self.backbone = BackboneFactory.produce(cfg)
        self.heads = nn.ModuleDict({
            'hm': nn.ModuleList([HourGlassHead(256, cfg.DB.NUM_CLASSES), HourGlassHead(256, cfg.DB.NUM_CLASSES)]),
            'wh': nn.ModuleList([HourGlassHead(256, 2), HourGlassHead(256, 2)]),
            'reg': nn.ModuleList([HourGlassHead(256, 2), HourGlassHead(256, 2)])
        })
        for head in self.heads['hm']:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        m.bias.data.fill_(-2.19)

    def forward(self, x):
        outs = self.backbone(x)
        outputs = []
        for i, out in enumerate(outs):
            head_out = {}
            for out_size in self.out_sizes:
                head_out[out_size] = {}
                for head in self.heads:
                    head_out[out_size][head] = self.heads[head][i](out)
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
        hm_loss = []
        wh_loss = []
        reg_loss = []
        for feat in feats:
            for out_size in feat:
                for head in feat[out_size]:
                    output = feat[out_size][head]
                    if head == 'hm':
                        output = _sigmoid(output)
                        hm_loss.append(self.crit[head](output, batch[out_size]['hm']).unsqueeze(0))
                    elif head == 'wh':
                        wh_loss.append(self.crit[head](output, batch[out_size]['reg_mask'], batch[out_size]['ind'], batch[out_size]['wh']).unsqueeze(0))
                    elif head == 'reg':
                        reg_loss.append(self.crit[head](output, batch[out_size]['reg_mask'], batch[out_size]['ind'], batch[out_size]['reg']).unsqueeze(0))
                    else:
                        raise TypeError
        losses = {'hm':torch.cat(hm_loss).mean(), 'wh':torch.cat(wh_loss).mean(), 'reg':torch.cat(reg_loss).mean()}
        loss = losses['hm'] + losses['wh'] * 0.1 + losses['reg']
        return loss, losses

class HourglassObjectDetection(BaseGraph):
    def __init__(self, cfg):
        super(HourglassObjectDetection, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)     
        self.loss_head = _LossHead()
        self.sub_models['loss'] = self.loss_head