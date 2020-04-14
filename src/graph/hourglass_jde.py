from src.graph import *
import math
from tools.utils import _sigmoid

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
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
            for head in self.heads:
                head_out[head] = self.heads[head][i](out)
            outputs.append(head_out)
        return outputs

class _LossHead(nn.Module):
    def __init__(self, cfg):
        super(_LossHead, self).__init__()
        self.crit = {}
        self.crit['hm'] = FocalLoss()  
        self.crit['wh'] = L1Loss()
        self.crit['reg'] = L1Loss()
        self.crit['embb'] = AMSoftmaxWithLoss()
        self.id_fc = AMSoftmaxClassiferHeadForJDE(128, cfg.REID.NUM_PERSON)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        
    def forward(self, feats, batch):
        hm_loss = []
        wh_loss = []
        reg_loss = []
        id_loss = []
        for feat in feats:
            for head in feat:
                output = feat[head]
                if head == 'hm':
                    output = _sigmoid(output)
                    hm_loss.append(self.crit[head](output, batch['hm']))
                elif head == 'wh':
                    wh_loss.append(self.crit[head](output, batch['reg_mask'], batch['ind'], batch['wh']))
                elif head == 'reg':
                    reg_loss.append(self.crit[head](output, batch['reg_mask'], batch['ind'], batch['reg']))
                elif head == 'embb':
                    cosine = self.id_fc(output, batch['reg_mask'], batch['ind'])
                    id_loss.append(self.cirt[head](cosine, batch['ids'][batch['reg_mask'] > 0]))
                else:
                    raise TypeError
        losses = {'hm':torch.mean(hm_loss) , 'wh':torch.mean(wh_loss), 'reg':torch.mean(reg_loss), 'embb':torch.mean(id_loss)}
        loss = torch.exp(-self.s_det) * (losses['hm'] + losses['wh'] * 0.1 + losses['reg']) + torch.exp(-self.s_id) * losses['embb'] + self.s_det + self.s_id
        return loss, losses

class HourglassJDE(BaseGraph):
    def __init__(self, cfg):
        super(HourglassJDE, self).__init__(cfg)        
    
    def build(self):
        self.model = _Model(self.cfg)     
        self.loss_head = _LossHead(self.cfg)
        self.sub_models['loss'] = self.loss_head
            
