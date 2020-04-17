import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, strides, stages_repeats, stages_out_channels, inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()
        self.stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self.stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, strides[0], 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        
        if strides[1] == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=strides[1], padding=1)
        else:
            self.maxpool = nn.BatchNorm2d(input_channels)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, stride, repeats, output_channels in zip(
                stage_names, strides[2:], stages_repeats, self.stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, stride)]
            for _ in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # output_channels = self.stage_out_channels[-1]
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(output_channels),
        #     nn.ReLU(inplace=True),
        # )


    def forward(self, x):
        # See note [TorchScript super()]
        stage_feats = []
        x = self.conv1(x)
        x = self.maxpool(x) # os 2
        x = self.stage2(x) # os 4
        stage_feats.append(x)
        x = self.stage3(x) # os 8
        stage_feats.append(x)
        x = self.stage4(x) # os 16
        stage_feats.append(x)
        # x = self.conv5(x) # os 16
        # stage_feats.append(x)
        return stage_feats

def shufflenetv2():
    return ShuffleNetV2([1, 2, 2, 2, 2], [4, 8, 4], [24, 116, 232, 464])

def shufflenetv2_low_resolution():
    return ShuffleNetV2([1, 1, 2, 2, 2], [4, 8, 4], [24, 116, 232, 464])