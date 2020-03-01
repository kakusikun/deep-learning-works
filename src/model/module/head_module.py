import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)        
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(in_channels, num_classes)   
    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class ReshapeHead(nn.Module):
    def __init__(self, in_channels, num_dim):   
        super(ReshapeHead, self).__init__()
        self.reshape = nn.Conv2d(
            in_channels,
            num_dim,
            1,
            stride=1,
            padding=0,
            bias=True)
    def forward(self, x):
        return self.reshape(x)