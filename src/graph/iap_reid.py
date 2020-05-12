from src.graph import *

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 
        self.head = IAPHead(cfg.MODEL.FEATSIZE, (16, 8), 256)
        self.iap_cosine_head = AMSoftmaxClassiferHead(256, cfg.REID.NUM_PERSON)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)[-1]
        embb = self.head(x)
        if self.iap_cosine_head.training:
            output = self.iap_cosine_head(embb)
        else:
            output = embb
        return output

class IAPReID(BaseGraph):
    def __init__(self, cfg):
        super(IAPReID, self).__init__(cfg)        

    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        self.crit['amsoftmax'] = AMSoftmaxWithLoss(s=30, m=0.35, relax=0.3)

        def loss_head(output, batch):
            _loss = self.crit['amsoftmax'](output, batch['pid'])
            losses = {
                'amsoftmax':_loss, 
            }
            loss = losses['amsoftmax']
            return loss, losses

        self.loss_head = loss_head
