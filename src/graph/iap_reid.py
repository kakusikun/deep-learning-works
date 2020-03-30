from src.graph import *

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 
        self.head = IAPHead(cfg.MODEL.FEATSIZE, (16, 8), 256)
        self.iap_cosine_head = AMSoftmaxClassiferHead(256, cfg.DB.NUM_CLASSES)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)[-1]
        embb = self.head(x)
        outputs = {
            'embb': embb,
            'cosine': self.iap_cosine_head(embb) if self.iap_cosine_head.training else None,
        }
        return outputs

class IAPReID(BaseGraph):
    def __init__(self, cfg):
        super(IAPReID, self).__init__(cfg)        

    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        self.crit['amsoftmax'] = AMSoftmaxWithLoss(s=30, m=0.35, relax=0.3)

        def loss_head(outputs, batch):
            _loss, logit = self.crit['amsoftmax'](outputs['cosine'], batch['pid'])
            losses = {
                'amsoftmax':_loss, 
            }
            loss = losses['amsoftmax']
            return loss, losses, logit

        self.loss_head = loss_head
