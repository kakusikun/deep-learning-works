from src.graph import *
import math
from tools.utils import _sigmoid

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        w, h = cfg.INPUT.SIZE
        self.out_sizes = [(w // s, h // s) for s in cfg.MODEL.STRIDES]
        self.backbone = BackboneFactory.produce(cfg)
        self.feature_extraction = biFPN(["i_0", "i_1", "i_2", "i_3"], self.backbone.stage_out_channels[-4:], [1,2,4,8], 64, n_layers=3, fpn=False, weighted_resize=False)
        self.heads = nn.ModuleDict({
            'hm': nn.ModuleList([HourGlassHead(64, cfg.DB.NUM_CLASSES) for _ in range(len(self.out_sizes))]),
            'wh': nn.ModuleList([HourGlassHead(64, 4) for _ in range(len(self.out_sizes))]),
            'reg': nn.ModuleList([HourGlassHead(64, 2) for _ in range(len(self.out_sizes))]),
        })

    def forward(self, x):
        outs = self.backbone(x)
        outs = self.feature_extraction(outs)[::-1]
        head_out = {}
        for i, out_size in enumerate(self.out_sizes):
            head_out[out_size] = {}
            for head in self.heads:
                head_out[out_size][head] = self.heads[head][i](outs[i])
        return head_out

class _LossHead(nn.Module):
    def __init__(self, cfg):
        super(_LossHead, self).__init__()
        self.crit = {}
        self.crit['hm'] = FocalLoss()  
        self.crit['iou'] = FsafCIOULoss()
        self.s_hm = nn.Parameter(torch.zeros(1))
        self.s_iou = nn.Parameter(torch.zeros(1))
        
    def forward(self, feats, batch):
        hm_loss = []
        iou_loss = []
        for out_size in feats:
            hm_loss.append(self.crit['hm'](feats[out_size]['hm'], batch[out_size]['hm']).unsqueeze(0))
            iou_loss.append(self.crit['iou'](feats[out_size]['wh'], feats[out_size]['reg'], batch[out_size]['mask'], batch[out_size]['ecount'], batch[out_size]['ind'], batch['bboxes']).unsqueeze(0))
        losses = {'hm':torch.cat(hm_loss).mean(), 'iou':torch.cat(iou_loss).mean()}
        loss = torch.exp(-self.s_hm) * losses['hm'] + torch.exp(-self.s_iou) * losses['iou'] + self.s_hm + self.s_iou
        uncertainty = {'s_hm': self.s_hm, 's_iou': self.s_iou}
        losses.update(uncertainty)
        return loss, losses

class ShuffleNetv2FSAF(BaseGraph):
    def __init__(self, cfg):
        super(ShuffleNetv2FSAF, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)     
        self.loss_head = _LossHead(self.cfg)
        self.sub_models['loss'] = self.loss_head
            
