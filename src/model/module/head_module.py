import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.module.base_module import ConvModule, SEModule, HSwish

class ClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassifierHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)        
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(in_channels, num_classes)   
    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_dim, num_layers=4, feat_size=64):   
        super(RegressionHead, self).__init__()
        features = []
        for _ in range(num_layers):
            features.append(
                ConvModule(
                    in_channels,
                    feat_size,
                    3,
                    stride=1,
                    padding=1,
                    activation='relu',
                    use_bn=False
                )
            )
            in_channels = feat_size
        self.features = nn.Sequential(*features)
        self.head = ConvModule(
            in_channels, num_dim, 3, stride=1, padding=1, activation='linear', use_bn=False)
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

class MobileNetv3ClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes, featc=1024):
        #TODO: recorrect
        super(MobileNetv3ClassifierHead, self).__init__()
        
        self._initialize_weights()

    def forward(self, x):
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ShuffleNetv2PlusClassifierHead(nn.Module):
    def __init__(self, in_channels, num_classes, featc=1024):
        super(ShuffleNetv2PlusClassifierHead, self).__init__()
        featc = int(featc * 0.75)
        self.v3_conv = ConvModule(in_channels, featc, 1, activation='hs')
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.v3_se = SEModule(featc)
        self.v3_fc = nn.Linear(featc, featc, bias=False)
        self.v3_hs = HSwish()
        self.dropout = nn.Dropout(0.2)
        self.v3_fc2 = nn.Linear(featc, num_classes, bias=False)

        self._initialize_weights()

    def forward(self, x):
        x = self.v3_conv(x)
        x = self.gap(x)
        x = self.v3_se(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.v3_fc(x)
        x = self.dropout(x)
        x = self.v3_fc2(x)
        return x
        
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class HourGlassHead(nn.Module):
    def __init__(self, in_channels, n_dim):
        super(HourGlassHead, self).__init__()    

        self.head = nn.Sequential(
            ConvModule(in_channels, 256, 1, use_bn=False),
            ConvModule(256, n_dim, 1, activation='linear', use_bn=False, bias=True)
        )
    
    def forward(self, x):
        return self.head(x)

class ReIDTrickHead(nn.Module):
    def __init__(self, in_channels, n_dim, use_gap=True):
        super(ReIDTrickHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1) if use_gap else None        
        self.BNNeck = nn.BatchNorm1d(in_channels)
        self.BNNeck.bias.requires_grad_(False)  # no shift
        self.BNNeck.apply(self.weights_init_kaiming)
        self.id_fc = nn.Linear(in_channels, n_dim, bias=False)        
        self.id_fc.apply(self.weights_init_classifier)

    def forward(self, x):
        if self.gap:
            x = self.gap(x)
            local_feat = x.view(x.size(0), -1)
        else:
            local_feat = x
        x = self.BNNeck(local_feat)
        global_feat = None
        if self.training:
            global_feat = self.id_fc(x)
        return x, local_feat, global_feat
    
    def weights_init_kaiming(self, module):
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

    def weights_init_classifier(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
class ReIDL2Head(nn.Module):
    def __init__(self, in_channels, n_pids):
        super(ReIDL2Head, self).__init__()
        self.id_fc = nn.Linear(in_channels, n_pids, bias=False)        
        self.id_fc.apply(self.weights_init_classifier)

    def forward(self, x):
        l2_feat = F.normalize(x)
        if self.training:
            id_feat = self.id_fc(x)
            return l2_feat, id_feat
        return l2_feat, None
    
    def weights_init_classifier(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class IAPHead(nn.Module):
    def __init__(self, in_channels, kernal_size, feat_dim):
        super(IAPHead, self).__init__()
        self.iap_GDConv = ConvModule(
            in_channels,
            in_channels,
            kernal_size,
            groups=in_channels,
            activation='linear',
            use_bn=False
        )
        self.iap_fc = nn.Linear(in_channels, feat_dim, bias=False)   

        self._init_params()

    def forward(self, x):
        x = self.iap_GDConv(x).view(x.size(0), -1)
        x = self.iap_fc(x)
        x = F.normalize(x)
        return x
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)