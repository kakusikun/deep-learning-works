from src.graph import *

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 
        self.head = IAPHead(cfg.MODEL.FEATSIZE, (16, 8), 256)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)[-1]
        x = self.head(x)
        outputs = {
            'embb': x,
        }
        return outputs

class IAPReID(BaseGraph):
    def __init__(self, cfg):
        super(IAPReID, self).__init__(cfg)        

    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        if self.use_gpu:
            self.crit['amsoftmax'] = AMSoftmaxWithLoss(256, self.cfg.DB.NUM_CLASSES, relax=0.3).cuda()
        else:
            self.crit['amsoftmax'] = AMSoftmaxWithLoss(256, self.cfg.DB.NUM_CLASSES, relax=0.3)

        def loss_head(outputs, batch):
            _loss, logit = self.crit['amsoftmax'](outputs['embb'], batch['pid'])
            losses = {
                'amsoftmax':_loss, 
            }
            loss = losses['amsoftmax']
            return loss, losses, logit

        self.loss_head = loss_head
