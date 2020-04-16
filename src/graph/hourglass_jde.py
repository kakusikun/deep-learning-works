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
            'reg': nn.ModuleList([HourGlassHead(256, 2), HourGlassHead(256, 2)]),
            'embb': nn.ModuleList([
                ConvModule(256, 128, 1, bias=True, activation='linear', use_bn=False),
                ConvModule(256, 128, 1, bias=True, activation='linear', use_bn=False)
            ])
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
    def __init__(self, cfg):
        super(_LossHead, self).__init__()
        self.crit = {}
        self.crit['hm'] = FocalLoss()  
        self.crit['wh'] = L1Loss()
        self.crit['reg'] = L1Loss()
        self.crit['embb'] = AMSoftmaxWithLoss(s=30, m=0.35, relax=0.3)
        self.id_fc = AMSoftmaxClassiferHeadForJDE(128, cfg.REID.NUM_PERSON)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        
    def forward(self, feats, batch):
        hm_loss = []
        wh_loss = []
        reg_loss = []
        id_loss = []
        logits = []
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
                    elif head == 'embb':
                        id_target = batch[out_size]['pids'][batch[out_size]['reg_mask'] > 0]
                        if len(id_target) > 0:
                            cosine = self.id_fc(output, batch[out_size]['reg_mask'], batch[out_size]['ind'])                            
                            _id_loss, logit = self.crit[head](cosine, id_target)
                        else:
                            _id_loss = torch.Tensor(0.0)
                            logit = None
                        id_loss.append(_id_loss.unsqueeze(0))
                        logits.append(logit)
                    else:
                        raise TypeError
        losses = {'hm':torch.cat(hm_loss).mean(), 'wh':torch.cat(wh_loss).mean(), 'reg':torch.cat(reg_loss).mean(), 'embb':torch.cat(id_loss).mean()}
        loss = torch.exp(-self.s_det) * (losses['hm'] + losses['wh'] * 0.1 + losses['reg']) + torch.exp(-self.s_id) * losses['embb'] + self.s_det + self.s_id
        return loss, losses, logits[1]

class HourglassJDE(BaseGraph):
    def __init__(self, cfg):
        super(HourglassJDE, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)     
        self.loss_head = _LossHead(self.cfg)
        self.sub_models['loss'] = self.loss_head
            
