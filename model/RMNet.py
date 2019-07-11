import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utility import Flatten

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernal=1, stride=1, padding=0, groups=1, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(kernal, kernal), stride=(stride, stride), padding=(padding, padding), groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernal=1, stride=1, padding=0, groups=1, bias=False, activation=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(kernal, kernal), stride=(stride, stride), padding=(padding, padding), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.activation = activation
        if self.activation:
            self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.elu(x)
        return x

class RMBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(RMBottleneck, self).__init__()
        self.conv1x1_in = ConvBlock(in_planes=in_planes, out_planes=out_planes // 4)
        if in_planes != out_planes:
            self.dwconv3x3 = ConvBlock(in_planes=out_planes // 4, out_planes=out_planes // 4, kernal=3, stride=2, padding=1, groups=out_planes // 4)
        else:
            self.dwconv3x3 = ConvBlock(in_planes=out_planes // 4, out_planes=out_planes // 4, kernal=3, padding=1, groups=out_planes // 4)
        self.conv1x1_out = ConvBlock(in_planes=out_planes // 4, out_planes=out_planes, activation=False)
        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                ConvBlock(in_planes=in_planes, out_planes=out_planes, activation=False)
            )
        self.elu = nn.ELU(inplace=True)
    def forward(self, x):
        out = self.conv1x1_in(x)
        out = self.dwconv3x3(out)
        out = self.conv1x1_out(out) + self.shortcut(x)
        out = self.elu(out)        
        return out


class RMNet(nn.Module):
    def __init__(self, b=[2,2,2], cifar10=False, reid=True, trick=False):
        super(RMNet, self).__init__()

        self.reid = reid
        self.trick = trick

        self.data_norm = nn.BatchNorm2d(3)
        if not cifar10:
            self.conv1 = ConvBlock(3, 32, kernal=3, stride=2, padding=1)
        else:
            self.conv1 = ConvBlock(3, 32, kernal=3, stride=1, padding=1)

        self.bottleneck_1 = self._make_layers(32, 64, b[0])
        self.bottleneck_2 = self._make_layers(64, 128, b[1])
        self.bottleneck_3 = self._make_layers(128, 256, b[2])
        self.bottleneck_4 = self._make_layers(256, 256, b[3])

        self.conv2 = ConvBlock(256, 512)
        
        
        if not self.trick:
            self.l_features = ConvBlock(512, 256, activation=False)
            self.g_features = ConvBlock(256, 256)
        else:
            self.l_features = ConvBlock(512, 256)

        self.flatten = Flatten()


    def _make_layers(self, in_planes, out_planes, block_size):
        RBs = []
        for i in range(block_size):
            if i != block_size-1:
                rb = RMBottleneck(in_planes, in_planes)
            else:
                rb = RMBottleneck(in_planes, out_planes)
            RBs.append(rb)
        return nn.Sequential(*RBs)  

    def forward(self, x):
        x = self.data_norm(x)

        x = self.conv1(x)

        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)

        x = F.adaptive_max_pool2d(x, 1)

        x = self.conv2(x)

        if self.reid:
            x = self.l_features(x)

            if not self.trick:
                local = F.normalize(self.flatten(x))
                x = self.g_features(x)
                glob = F.normalize(self.flatten(x))

                return local, glob
            else:
                return x
        else:
            x = self.flatten(x)
            return x

if __name__ == '__main__':
    model = RMNet(b=[4,8,10,11], cifar10=False, reid=True)
    dummyInput = torch.ones(1,3,384,128)
    model.eval()
    with torch.no_grad():
        output = model(dummyInput)
