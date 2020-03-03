import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.module.base_module import ConvModule

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