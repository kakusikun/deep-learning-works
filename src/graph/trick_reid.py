from src.graph import *

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 
        self.head = ReIDTrickHead(cfg.MODEL.FEATSIZE, cfg.DB.NUM_CLASSES)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)[-1]
        x1, x2, x3 = self.head(x)
        outputs = {
            'neck': x1,
            'local': x2,
            'global': x3
        }
        return outputs

class TrickReID(BaseGraph):
    def __init__(self, cfg):
        super(TrickReID, self).__init__(cfg)        

    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        self.crit['cels'] = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)
        self.crit['triplet'] = TripletLoss(0.3)
        self.crit['center'] = CenterLoss(self.cfg.MODEL.FEATSIZE, self.cfg.DB.NUM_CLASSES) 
        self.sub_models['center'] = self.crit['center']
                        
        def loss_head(outputs, batch):
            assert outputs['global'] is not None
            losses = {
                'cels':self.crit['cels'](outputs['global'], batch['pid']), 
                'triplet':self.crit['triplet'](outputs['local'], batch['pid'])[0], 
                'center':self.crit['center'](outputs['local'], batch['pid'])
            }
            loss = losses['cels'] + losses['triplet'] + self.cfg.REID.CENTER_LOSS_WEIGHT * losses['center']
            return loss, losses

        self.loss_head = loss_head
