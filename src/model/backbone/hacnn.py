import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.module.base_module import ConvModule, GeM, SEModule
from src.model.module.hacnn_module import HABlock

class ShuffleBlock(nn.Module):
    def __init__(self, inc, midc, ouc, ksize, stride, activation, useSE, mode, affine=True):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        pad = ksize // 2
        inc = inc // 2

        if mode == 'v2':
            branch_main = [
                ConvModule(inc, midc, 1, activation=activation, affine=affine),
                ConvModule(midc, midc, ksize, stride=stride, padding=pad, groups=midc, activation='linear', affine=affine),
                ConvModule(midc, ouc - inc, 1, activation=activation, affine=affine),
            ]
        elif mode == 'xception':
            assert ksize == 3
            branch_main = [
                ConvModule(inc, inc, 3, stride=stride, padding=1, groups=inc, activation='linear', affine=affine),
                ConvModule(inc, midc, 1, activation=activation, affine=affine),
                ConvModule(midc, midc, 3, stride=1, padding=1, groups=midc, activation='linear', affine=affine),
                ConvModule(midc, midc, 1, activation=activation, affine=affine),
                ConvModule(midc, midc, 3, stride=1, padding=1, groups=midc, activation='linear', affine=affine),
                ConvModule(midc, ouc - inc, 1, activation=activation, affine=affine),
            ]
        else:
            raise TypeError
        
        if activation == 'relu':
            assert useSE == False
        else:
            if useSE:
                branch_main.append(SEModule(ouc - inc))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            self.branch_proj = nn.Sequential(
                ConvModule(inc, inc, ksize, stride=stride, padding=pad, groups=inc, activation='linear', affine=affine),
                ConvModule(inc, inc, 1, activation=activation, affine=affine),
            )
        else:
            self.branch_proj = None

    def forward(self, x):
        x_proj, x = channel_shuffle(x)
        if self.stride==1:
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class ShuffleA(nn.Module):
    def __init__(self, inc, ouc):
        super(ShuffleA, self).__init__()
        midc = ouc // 2
        self.block = ShuffleBlock(inc, midc, ouc, 1, 
            stride=1, 
            activation='relu',
            useSE=False, 
            mode='v2'
        )
    def forward(self, x):
        return self.block(x)

class ShuffleB(nn.Module):
    def __init__(self, inc, ouc, stride):
        super(ShuffleB, self).__init__()
        midc = ouc // 2
        self.block = ShuffleBlock(inc, midc, ouc, 3,
            stride=stride,
            activation='relu',
            useSE=False,
            mode='v2'
        )
    def forward(self, x):
        return self.block(x)

class HACNN(nn.Module):
    """Harmonious Attention Convolutional Neural Network.
    Reference:
        Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.
    Public keys:
        - ``hacnn``: HACNN.
    """

    # Args:
    #    num_classes (int): number of classes to predict
    #    nchannels (list): number of channels AFTER concatenation
    #    feat_dim (int): feature dimension for a single stream
    #    learn_region (bool): whether to learn region features (i.e. local branch)

    def __init__(
        self,
        stage_repeats,
        stage_out_channels,
        **kwargs
    ):
        super(HACNN, self).__init__()
        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels
        in_channels = self.stage_out_channels[0]
        self.conv = ConvModule(3, in_channels, 3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        self.has = nn.ModuleList()

        for stage_i in range(len(self.stage_repeats)):
            stage = []
            out_channels = self.stage_out_channels[stage_i+1]
            stage.append(ShuffleB(in_channels, out_channels, 1))
            for _ in range(self.stage_repeats[stage_i]):
                stage.append(ShuffleA(out_channels, out_channels))
            stage.append(ShuffleB(out_channels, out_channels, 2))
            self.has.append(HABlock(out_channels))
            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels

        feat_dim = self.stage_out_channels[-1]
        self.global_fc = nn.Sequential(
            nn.Linear(out_channels, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        self.pooling = GeM()
        self._init_params()

    def forward(self, x):
        assert x.size(2) == 160 and x.size(3) == 64, \
            'Input size does not match, expected (160, 64) but got ({}, {})'.format(x.size(2), x.size(3))
        stages = []
        x = self.conv(x)

        for stage_i in range(len(self.stage_repeats)):
            x_stage = self.stages[stage_i](x)
            x_attn, x_theta = self.has[stage_i](x_stage)
            stages.append((x, x_theta))
            x = x_stage * x_attn
        
        x_global = self.pooling(x).view(x.size(0), -1)
        x_global = self.global_fc(x_global)
        stages.append((x_global, None))
        return stages

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
def hacnn():
    model = HACNN([7, 10, 7], [32, 128, 256, 384, 512])
    return model
    

if __name__ == "__main__":
    from src.model.module.base_module import HarmAttn
    model = HACNN([7, 10, 7], [32, 128, 256, 384, 512])
    ha = HarmAttn(ShuffleB, 4, 32, [128, 256, 384], 512)
    x = torch.rand(2,3,160,64)

    output = model(x)
    global_feat, _ = output[-1]
    local_feat = ha(output[:3])
    print(global_feat.shape)