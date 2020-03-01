from src.graph import *
from src.model.module import ClassificationHead

class _Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #TODO: migrate MODEl to GRAPH
        self.backbone = BackboneFactory.produce(cfg)
        self.head = ClassificationHead(self.backbone.last_channel, cfg.DB.NUM_CLASSES)
    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.head(x)
        return x

class SimpleClassifier(BaseGraph):
    def __init__(self, cfg):
        super().__init__(cfg)  

    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        self.crit['ce'] = nn.CrossEntropyLoss()

        def loss_head(feat, batch):
            losses = {'ce':self.crit['ce'](feat, batch['target'])}
            loss = losses['ce']
            return loss, losses
        self.loss_head = loss_head


def weights_init_classifier(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            

