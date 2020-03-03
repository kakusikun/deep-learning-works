from src.graph import *

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 

        # TODO: move to head module
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#
        # self.gap = nn.AdaptiveAvgPool2d(1)        
        # self.BNNeck = nn.BatchNorm1d(self.backbone.feature_dim)
        # self.BNNeck.bias.requires_grad_(False)  # no shift
        # self.BNNeck.apply(weights_init_kaiming)

        # self.id_fc = nn.Linear(self.backbone.feature_dim, cfg.DB.NUM_CLASSES, bias=False)        
        # self.id_fc.apply(weights_init_classifier)
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        feat = self.backbone(x)

        # TODO: move to head module
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#
        # x = self.gap(feat)
        # local_feat = x.view(x.size(0), -1)
        # x = self.BNNeck(local_feat)
        # #  x = x.view(x.size(0), -1)        
        # if not self.training:
        #     return x      
        # global_feat = self.id_fc(x)  
        # return local_feat, global_feat
        # ------------------------------------------------------------------------#
        # ------------------------------------------------------------------------#




class ReIDWithTrick(BaseGraph):
    def __init__(self, cfg):
        super(ReIDWithTrick, self).__init__(cfg)        

        if cfg.TASK == "reid":
            self._make_model()
            self._make_loss()
        else:
            logger.info("Task {} is not supported".format(cfg.TASK))  
            sys.exit(1)

                    
                        
    def _make_model(self):
        self.model = Model(self.cfg)

    def _make_loss(self):
        self.crit = {}
        self.crit['cels'] = CrossEntropyLossLS(self.cfg.DB.NUM_CLASSES)
        self.crit['triplet'] = TripletLoss()
        self.crit['center'] = CenterLoss(512, self.cfg.DB.NUM_CLASSES) 
        self.submodels['center'] = self.crit['center']

        def loss_func(l_feat, g_feat, target):
            losses = {'cels':self.crit['cels'](g_feat, target), 
                         'triplet':self.crit['triplet'](l_feat, target)[0], 
                         'center':self.crit['center'](l_feat, target)}

            loss = losses['cels'] + losses['triplet'] + self.cfg.SOLVER.CENTER_LOSS_WEIGHT * losses['center']
            return loss, losses

        self.loss_func = loss_func

def weights_init_kaiming(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
            

