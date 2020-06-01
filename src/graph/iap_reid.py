from src.graph import *
import torchvision.transforms.functional as TF


class IAPReID(BaseGraph):
    def __init__(self, cfg):
        super(IAPReID, self).__init__(cfg)        

    def build(self):
        self.model = _Model(self.cfg)
        self.crit = {}
        self.crit['amsoftmax'] = AMSoftmaxWithLoss(s=30, m=0.35, relax=0.0)

        def loss_head(output, batch):
            _loss = self.crit['amsoftmax'](output, batch['pid'])
            losses = {
                'amsoftmax':_loss, 
            }
            loss = losses['amsoftmax']
            return loss, losses

        self.loss_head = loss_head

class DualNormIAPReID(BaseGraph):
    def __init__(self, cfg):
        super(DualNormIAPReID, self).__init__(cfg)        

    def build(self):
        self.model = _Model2(self.cfg)
        self.torchscript_model = TorchSciptModel(self.cfg)
        self.crit = {}
        self.crit['cels'] = CrossEntropyLossLS(self.cfg.REID.NUM_PERSON)

        def loss_head(output, batch):
            _loss = self.crit['cels'](output, batch['pid'])
            losses = {
                'cels':_loss, 
            }
            loss = losses['cels']
            return loss, losses

        self.loss_head = loss_head

class _Model(nn.Module):
    def __init__(self, cfg):
        super(_Model, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 
        self.head = IAPHead(cfg.MODEL.FEATSIZE, (16, 8), 256)
        self.iap_cosine_head = AMSoftmaxClassiferHead(256, cfg.REID.NUM_PERSON)
    
    def forward(self, x):
        # use trick: BNNeck, feature before BNNeck to triplet GAP and feature w/o fc forward in backbone
        x = self.backbone(x)
        embb = self.head(x)
        if self.iap_cosine_head.training:
            output = self.iap_cosine_head(embb)
        else:
            output = embb
        return output

class _Model2(nn.Module):
    def __init__(self, cfg):
        super(_Model2, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 
        self.iap_trick_head = ReIDTrickHead(cfg.MODEL.FEATSIZE, n_dim=cfg.REID.NUM_PERSON, kernal_size=(16, 8))
        # self.iap_cosine_head = AMSoftmaxClassiferHead(cfg.MODEL.FEATSIZE, cfg.REID.NUM_PERSON)
    
    def forward(self, x):
        x = self.backbone(x)
        y = self.iap_trick_head(x)
        # if self.iap_cosine_head.training:
        #     return self.iap_cosine_head(y)
        # else:
        return y

class StaticReIDTrickHead(nn.Module):
    def __init__(self, in_channels, kernal_size):
        super(StaticReIDTrickHead, self).__init__()
        self.gap = ConvModule(
            in_channels,
            in_channels,
            kernal_size,
            groups=in_channels,
            activation='linear',
            use_bn=False
        )
        self.BNNeck = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.gap(x)
        x = self.BNNeck(x)
        x = x.view(-1, x.size(1))
        return x

class TorchSciptModel(nn.Module):
    def __init__(self, cfg):
        super(TorchSciptModel, self).__init__()
        self.backbone = BackboneFactory.produce(cfg) 
        self.iap_trick_head = StaticReIDTrickHead(cfg.MODEL.FEATSIZE, kernal_size=(16, 8))
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def preprocess(self, x):
        x = TF.normalize(x, self.mean, self.std).unsqueeze(0)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        y = self.iap_trick_head(x)
        return y


