import torch.nn as nn
import torch.nn.functional as F
import math

NormFunc = nn.BatchNorm2d

class SEBlock(nn.Module):
  def __init__(self, channels, reduction = 16):
    super(SEBlock, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc1 = nn.Conv2d(
      channels, channels // reduction, kernel_size=1, padding=0, bias=False)
    self.pelu = nn.PReLU()
    self.fc2 = nn.Conv2d(
      channels // reduction, channels, kernel_size=1, padding=0, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    module_input = x
    # C: channels, r: reduction

    x = self.avg_pool(x) 
    # => 1 x 1 x C

    x = self.fc1(x)
    # => 1 x 1 x (C / r)

    x = self.pelu(x)

    x = self.fc2(x)
    # => 1 x 1 x C

    x = self.sigmoid(x)

    return module_input * x  # C_input_i = C_input_i * x_i => scale the channel 


class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, bottle_neck=False, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        if bottle_neck:
          self.norm1 = NormFunc(inplanes)
          self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
          self.norm2 = NormFunc(planes)
          self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
          self.norm3 = NormFunc(planes)
          self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
          self.norm4 = NormFunc(planes * 4)
          self.prelu = nn.PReLU()
        else:
          self.norm1 = NormFunc(inplanes)
          self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
          self.norm2 = NormFunc(planes )
          self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bias=False)
          self.norm3 = NormFunc(planes)
          self.prelu = nn.PReLU()
        self.bottle_neck = bottle_neck
        self.downsample = downsample
        self.use_se = use_se
        if self.use_se:
          if bottle_neck:
            self.se = SEBlock(planes * 4)
          else:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        if self.bottle_neck:
          out = self.norm1(x)
          out = self.conv1(out)
          out = self.norm2(out)
          out = self.prelu(out)

          out = self.conv2(out)
          out = self.norm3(out)
          out = self.prelu(out)
          out = self.conv3(out)
          out = self.norm4(out)
        else:
          out = self.norm1(x)
          out = self.conv1(out)
          out = self.norm2(out)
          out = self.prelu(out)

          out = self.conv2(out)
          out = self.norm3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

