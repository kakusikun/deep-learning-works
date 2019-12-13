import torch
import torch.nn as nn
import torch.nn.functional as F
from manager.utility import Flatten

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernal=1, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(kernal, kernal), stride=(stride, stride), padding=(padding, padding), groups=1)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Lite3x3(nn.Module):
    def __init__(self, in_planes):
        super(Lite3x3, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=(1,1), stride=(1,1), groups=1, padding=(0,0), bias=False)
        self.dwconv3x3 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, groups=in_planes, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.bn = nn.BatchNorm2d(num_features=in_planes)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.dwconv3x3(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class LiteBlock(nn.Module):
    def __init__(self, in_planes, blockSize):
        super(LiteBlock, self).__init__()        
        stream = []
        for _ in range(blockSize):
            LiteUnit = Lite3x3(in_planes)
            stream.append(LiteUnit)
        self.stream = nn.Sequential(*stream)

    def forward(self, x):
        return self.stream(x)

class AggregationGate(nn.Module):
    def __init__(self, in_planes, r=1):
        super(AggregationGate, self).__init__()
        self.fc1 = nn.Linear(in_planes, in_planes // r)
        self.fc2 = nn.Linear(in_planes // r, in_planes)
        self.flatten = Flatten()

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), -1, 1, 1)
        return x

class SEAggregationGate(nn.Module):
    def __init__(self, in_planes, r=1):
        super(SEAggregationGate, self).__init__()
        self.fc1 = nn.Linear(in_planes, in_planes // r)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(in_planes // r, in_planes)
        self.flatten = Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(x.size(0), -1, 1, 1)
        return x

class OmniBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, shared=False, r=1, use_se=False):
        super(OmniBottleneck, self).__init__()
        self.shared = shared
        
        middle_planes = in_planes if r == 1 else r

        self.conv1 = ConvBlock(in_planes, middle_planes)
        if not self.shared:
            self.stream1 = LiteBlock(middle_planes, 1)
            self.stream2 = LiteBlock(middle_planes, 2)
            self.stream3 = LiteBlock(middle_planes, 3)
            self.stream4 = LiteBlock(middle_planes, 4)
        else:
            self.stream1 = LiteBlock(middle_planes, 1)
            self.stream2 = LiteBlock(middle_planes, 1)
            self.stream3 = LiteBlock(middle_planes, 1)
            self.stream4 = LiteBlock(middle_planes, 1)

        if use_se:
            self.AG = SEAggregationGate(middle_planes, r=16)
        else:
            self.AG = AggregationGate(middle_planes, r=16)

        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                ConvBlock(in_planes, out_planes)
            )

        self.conv2 = ConvBlock(middle_planes, out_planes)

        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)

        if not self.shared:
            out1 = self.stream1(out)
            out2 = self.stream2(out)
            out3 = self.stream3(out)
            out4 = self.stream4(out)
        else:
            out1 = self.stream1(out)
            out2 = self.stream2(out1)
            out3 = self.stream3(out2)
            out4 = self.stream4(out3)

        onmi_out1 = self.AG(out1)
        onmi_out2 = self.AG(out2)
        onmi_out3 = self.AG(out3)
        onmi_out4 = self.AG(out4)       

        out1 = out1 * onmi_out1
        out2 = out2 * onmi_out2
        out3 = out3 * onmi_out3
        out4 = out4 * onmi_out4 

        out = out1 + out2 + out3 + out4

        out = self.conv2(out) + self.shortcut(x)

        out = self.prelu(out)

        return out


class OSNet(nn.Module):
    def __init__(self, widthMultiplier=1.0, shared=False, b=[2,2,2], r=[1,1,1], cifar10=False, use_se=False):
        super(OSNet, self).__init__()
        self.shared = shared
        self.cifar10 = cifar10
        self.use_se = use_se

        if self.cifar10:
            self.conv1 = nn.Sequential(ConvBlock(3, 64, kernal=3, stride=1, padding=1),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:                                   
            self.conv1 = nn.Sequential(ConvBlock(3, 64, kernal=7, stride=2, padding=3),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self._make_layers(64, 256, b[0], r[0])

        self.transition1 = nn.Sequential(ConvBlock(256, 256),
                                         nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv3 = self._make_layers(256, 384, b[1], r[1])

        self.transition2 = nn.Sequential(ConvBlock(384, 384),
                                         nn.AvgPool2d(kernel_size=2, stride=2))

        self.conv4 = self._make_layers(384, 512, b[2], r[2])

        self.conv5 = ConvBlock(512, 512)

        self.flatten = Flatten()

        self.fc = nn.Linear(512, 512)

    def _make_layers(self, in_planes, out_planes, blockSize, reduction):
        OBs = []
        for i in range(blockSize):
            if i != blockSize-1:
                ob = OmniBottleneck(in_planes, in_planes, self.shared, reduction, self.use_se)
            else:
                ob = OmniBottleneck(in_planes, out_planes, self.shared, reduction, self.use_se)
            OBs.append(ob)
        return nn.Sequential(*OBs)  

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.transition1(x)

        x = self.conv3(x)

        x = self.transition2(x)

        x = self.conv4(x)

        x = self.conv5(x)

        x = F.adaptive_avg_pool2d(x, 1)

        x = self.flatten(x)

        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = OSNet(r=[64,96,128], b=[2,2,2])
    dummyInput = torch.ones(1,3,256,128)
    model.eval()
    with torch.no_grad():
        output = model(dummyInput)