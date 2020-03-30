from src.graph import *

class _Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = BackboneFactory.produce(cfg)
        self.head = ShuffleNetv2PlusClassifierHead(cfg.MODEL.FEATSIZE, cfg.DB.NUM_CLASSES)
    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.head(x)
        return x

class ShuffleNetV2PlusClassifier(BaseGraph):
    def __init__(self, cfg):
        super().__init__(cfg)  

    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        self.crit['cels'] = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)

        def loss_head(feat, batch):
            losses = {'cels':self.crit['cels'](feat, batch['target'])}
            loss = losses['cels']
            return loss, losses
        self.loss_head = loss_head
          

