import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    """
    (standard) Conv => BN => ReLU
    (linear)   Conv => BN
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, linear=False):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride,
            padding=padding, 
            bias=False, 
            groups=groups)
        
        self.bn = nn.BatchNorm2d(out_channels)
        if not linear:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class InversedDepthwiseSeparable(nn.Module):
    """
    Inversed Depthwise Separable
    Conv 1x1 => dwConv => BN => ReLU
    """    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(InversedDepthwiseSeparable, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.conv2 = ConvModule(
            out_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=kernel_size // 2, 
            groups=out_channels
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DepthwiseSeparable(nn.Module):
    """
    Inversed Depthwise Separable
     dwConv => Conv 1x1 => BN => ReLU
    """    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(DepthwiseSeparable, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size, 
            stride=stride, 
            padding=kernel_size // 2, 
            bias=False,
            groups=in_channels
        )
        self.conv2 = ConvModule(
            in_channels, 
            out_channels, 
            1, 
            stride=1, 
            padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SEModule(nn.Module):
    """
    input, N x C x H x W
        => GAP 
        => Conv channel discounted 
        => ReLU 
        => FC channel restored 
        => hswish * input

    input, N x C
        => GAP 
        => Conv channel discounted 
        => ReLU 
        => FC channel restored 
        => hswish * input
    
    """
	def __init__(self, inplanes, isTensor=True):
		super(SEModule, self).__init__()
		if isTensor:
			# if the input is (N, C, H, W)
			self.se = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(inplanes, inplanes // 4, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Conv2d(inplanes // 4, inplanes, kernel_size=1, stride=1, bias=False),
			)
		else:
			# if the input is (N, C)
			self.se = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Linear(inplanes, inplanes // 4, bias=False),
				nn.BatchNorm1d(inplanes // 4),
				nn.ReLU(inplace=True),
				nn.Linear(inplanes // 4, inplanes, bias=False),
			)

	def forward(self, x):
		atten = self.se(x)
		atten = torch.clamp(atten + 3, 0, 6) / 6
		return x * atten

class Res2NetStem(nn.Module):
    """
    Conv 3x3xs2 => Conv 3x3 => Conv 3x3 => MAX 3x3xs2
    """
    def __init__(self, in_channels, out_channels):
        super(Res2NetStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                out_channels // 2,
                out_channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),        
            nn.MaxPool2d(
                3,
                stride=2,
                padding=1
            )   
        )
    def forward(self, x):
        return self.stem(x)

class DropPath(nn.Module):
    def __init__(self, p=0.2):
        """
        Drop path with probability.

        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), x.size(1), 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask
            
        return x

class DropChannel(nn.Module):
    def __init__(self, p=0.2):
        """
        Drop path with probability.

        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), x.size(1), 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask
            
        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class FusedNormalization(nn.Module):
    def __init__(self, size, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.rand(size))
        self.relu = nn.ReLU()
        self.eps = eps
    def forward(self, p):
        w = self.relu(self.w)
        w.div_(w.sum() + self.eps)
        fused = 0.0
        for wi, pi in zip(w, p):
            fused += wi*pi
        return fused
        
class biFPNLayer(nn.Module):
    def __init__(self, feat_size, level=3):
        super().__init__()
        self.p_lat1s = nn.ModuleList()
        self.p_lat2s = nn.ModuleList()
        self.p_ups = nn.ModuleList()
        self.p_downs = nn.ModuleList()
        self.p_w_td_adds = nn.ModuleList()
        self.p_w_bu_adds = nn.ModuleList()

        for _ in range(level-1):
            self.p_lat1s.append(DepthwiseSeparable(feat_size, feat_size, 1))
            self.p_lat2s.append(DepthwiseSeparable(feat_size, feat_size, 1))
            self.p_ups.append(nn.Upsample(scale_factor=2))
            self.p_w_td_adds.append(FusedNormalization(2))
            self.p_downs.append(nn.Upsample(scale_factor=0.5))
            self.p_w_bu_adds.append(FusedNormalization(3))
    
    def forward(self, ps): # high to low level, e.g., P7 -> P6 -> P5 ...
        level = len(ps)
        p_tds = [ps[0]] # high to low level
        for i in range(level-1):
            p_tds.append(
                self.p_lat1s[i](
                    self.p_w_td_adds[i](
                        [ps[i+1], self.p_ups[i](p_tds[i])]
                    )
                )
            )
        p_os = [p_tds[-1]] # high to low level
        for i in range(level-1):
            p_os.insert(0,
                self.p_lat2s[i](
                    self.p_w_bu_adds[i](
                        [ps[level-i-2], p_tds[level-i-2], self.p_downs[i](p_os[0])]
                    )
                )
            )
        return p_os
        

class biFPN(nn.Module):
    def __init__(self, in_feat_sizes, out_feat_size, level=3, num_layers=2):
        super().__init__()
        assert len(in_feat_sizes) == level
        self.level = level
        self.p_lats = nn.ModuleList()
        for i, in_feat_size in zip(range(level), in_feat_sizes):
            if i <= 2: 
                self.p_lats.append(nn.Conv2d(in_feat_size, out_feat_size, 1, stride=1, padding=0))
            elif i == 3:
                self.p_lats.append(nn.Conv2d(in_feat_sizes[-1], out_feat_size, 1, stride=1, padding=0))
            elif i > 3:
                self.p_lats.append(nn.Conv2d(out_feat_size, out_feat_size, 1, stride=1, padding=0))
        
        biFPNLayers = []
        for _ in range(num_layers):
            biFPNLayers.append(biFPNLayer(out_feat_size, level))
        self.biFPNLayers = nn.Sequential(*biFPNLayers)

    def forward(self, inputs): # low to high level, e.g., P3 -> P4 -> P5 ...
        ps = [] # high to low level, e.g., P7 -> P6 -> P5 ...
        for i in range(self.level):
            if i <= 2:
                ps.insert(0, self.p_lats[i](inputs[i]))               
            elif i == 3:
                ps.insert(0, self.p_lats[i](inputs[i-1]))
            else:
                ps.insert(0, self.p_lats[i](ps[i-1]))
        
        return self.biFPNLayers(ps)

if __name__ == '__main__':
    a = torch.rand([1,2,16,16])
    b = torch.rand([1,3,8,8])
    c = torch.rand([1,4,4,4])

    model = biFPN([2,3,4], 5, num_layers=2)
    model([a,b,c])

