from src.graph import *
from collections import OrderedDict
from src.model.module import ClassificationHead


class _Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #TODO: migrate MODEl to GRAPH
        self.backbone = BackboneFactory.produce(cfg.MODEL.BACKBONE)(cfg)
        self.head = ClassificationHead(self.backbone.feat_dim, cfg.DB.NUM_CLASSES)
    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.head(x)
        return x

class Classifier(BaseGraph):
    def __init__(self, cfg):
        super().__init__(cfg)  

    def build(self):
        if self.cfg.TASK == "classification":            
            self.model = _Model(self.cfg)
            self.crit = {}
            self.crit['ce'] = nn.CrossEntropyLoss()

            def loss_head(feat, batch):
                losses = {'ce':self.crit['ce'](feat, batch['target'])}
                loss = losses['ce']
                return loss, losses
            self.loss_head = loss_head
        else:
            logger.info("Task {} is not supported".format(self.cfg.TASK))  
            sys.exit(1)


def weights_init_classifier(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            

