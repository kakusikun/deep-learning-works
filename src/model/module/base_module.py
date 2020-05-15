import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvModule(nn.Module):
    """
    (#activation)
    (relu) Conv => BN (use_bn) => ReLU
    (hs)   Conv => BN (use_bn) => HSwish
    (linear)   Conv => BN (use_bn)
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, groups=1, bias=False,
        activation='relu', 
        use_bn=True, affine=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride,
            padding=padding, 
            bias=bias, 
            groups=groups)
        
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=affine)
        else:
            self.bn = None

        if activation == 'linear':
            self.activation = None
        else:
            if activation == 'relu':
                self.activation = nn.ReLU(inplace=True)
            elif activation == 'hs':
                self.activation = HSwish()
            else:
                raise TypeError            

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
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

class ECAModule(nn.Module):
    """
    Reference : https://arxiv.org/abs/1910.03151
    input, N x C x H x W
        => GAP, N x C x 1 x 1
        => N X 1 x C x 1
        => Conv, N X 1 x C x 1
        => N x C x 1 x 1
        =< Sigmoid
    """
    def __init__(self, inplanes, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((math.log(inplanes, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        atten = self.gap(x)
        atten = self.conv(atten.transpose(1,2))
        atten = self.sigmoid(atten.transpose(1,2))
        return x * atten.expand_as(x)

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
    def forward(self, x, *args, **kwargs):
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
    '''
            ps                                         p_tds                                                                         p_os
            ---------------------------------------------------------------------------------------------------------------------------------------
    high    p_level ---------------------------------> p_tds_level -------> w+(p_level, p_tds_level, d(p_tds_(level-1)))  --- f ---> - p_os_level
            .
            .
            .
            p_1     ------> w+(p_1, u(p_2)) --- f ---> p_tds_1     -------> w+(p_1, p_tds_1, d(p_0)))                     --- f ---> p_os_1            
    low     p_0     ------> w+(p_0, u(p_1)) --- f ---> p_tds_0     ----------------------------------------------------------------> p_os_0
    '''
    def __init__(self, feat_size, oss, fpn=False, weighted_resize=False):
        super().__init__()
        self.weighted_resize = weighted_resize
        self.fpn = fpn
        self.p_lat1s = nn.ModuleList()
        self.p_lat2s = nn.ModuleList()
        self.p_ups = nn.ModuleList()
        self.p_downs = nn.ModuleList()
        self.p_w_td_adds = nn.ModuleList()
        self.p_w_bu_adds = nn.ModuleList()

        for i in range(len(oss)-1):
            self.p_lat1s.append(DepthwiseSeparable(feat_size, feat_size, 1))
            self.p_lat2s.append(DepthwiseSeparable(feat_size, feat_size, 1))
            self.p_w_td_adds.append(FusedNormalization(2))
            self.p_w_bu_adds.append(FusedNormalization(3))
            if weighted_resize:
                self.p_ups.append(nn.ConvTranspose2d(feat_size, feat_size, 4, stride=2, padding=1) if oss[len(oss)-1-i] // 2 == oss[len(oss)-1-i-1] else nn.Sequential())
                self.p_downs.append(nn.Conv2d(feat_size, feat_size, 3, stride=2, padding=1) if oss[i] * 2 == oss[i+1] else nn.Sequential())
            else:
                self.p_ups.append(nn.Upsample(scale_factor=2) if oss[len(oss)-1-i] // 2 == oss[len(oss)-1-i-1] else nn.Sequential())
                self.p_downs.append(nn.Upsample(scale_factor=0.5) if oss[i] * 2 == oss[i+1] else nn.Sequential())
        
        if fpn:
            for i in range(len(oss)-1):
                if weighted_resize:
                    self.p_ups.append(nn.ConvTranspose2d((i+1)*feat_size, (i+1)*feat_size, 4, stride=2, padding=1) if oss[len(oss)-1-i] // 2 == oss[len(oss)-1-i-1] else nn.Sequential())
                else:
                    self.p_ups.append(nn.Upsample(scale_factor=2) if oss[len(oss)-1-i] // 2 == oss[len(oss)-1-i-1] else nn.Sequential())

    
    def forward(self, ps): # high to low level, e.g., P7 -> P6 -> P5 ...
        level = len(ps)
        p_tds = [ps[0]] 
        for i in range(level-1): # from high to low level
            p_tds.append(
                self.p_lat1s[i](
                    self.p_w_td_adds[i](
                        [ps[i+1], self.p_ups[i](p_tds[i])]
                    )
                )
            )
        p_os = [p_tds[-1]] 
        for i in range(level-1): # from low to high level
            p_os.insert(0,
                self.p_lat2s[i](
                    self.p_w_bu_adds[i](
                        [ps[level-i-2], p_tds[level-i-2], self.p_downs[i](p_os[0])]
                    )
                )
            )

        if self.fpn:
            fpn_feat = p_os[0]
            for i in range(level-1):
                fpn_feat = torch.cat((p_os[i+1], self.p_ups[level-1+i](fpn_feat)), dim=1)
            return fpn_feat
        return p_os # from high to low level

class biFPN(nn.Module):
    '''
    Args:
        config (list): 
            list of input feature type to be extracted by 
            lateral convolutional layer.
            For example, ["i_0", "i_1", "i_2", "i_2", "p_3"]
                where A_B,
                A: either "i" or "p", "i" means the feature is from input and "p" is from 
                    lateral convolutional  layer.
                B: the index of inputs or feature of lateral convolutional layer.
            In this example, 
            index 0 of input (i) -> lateral conv -> index 0 of lateral feature (p)
            index 1 of input (i) -> lateral conv -> index 1 of lateral feature (p)
            index 2 of input (i) -> lateral conv -> index 2 of lateral feature (p)
            index 2 of input (i) -> lateral conv -> index 3 of lateral feature (p)
            index 3 of lateral feature (p) -> lateral conv -> index 4 of lateral feature (p)

        incs (list):
            list of input channels for feature in config with "i" prefixed.

        oss (list):
            list of output stride for feature in config.
            
        ouc (int): 
            output channel for lateral convolutional layer that extracts feature from inputs

        n_layers (int): 
            number of top-down and bottom-up combination (=> biFPNLayer)

        fpn (bool): 
            whether upsample the high-level feature and concat to one output

        weighted_resize (bool):
            down- and upsampling is nn.Upsample or nn.Conv2d and nn.ConvTranspose2d respectively
            
    '''
    def __init__(self, configs, incs, oss, ouc, n_layers=2, fpn=False, weighted_resize=False):
        super().__init__()
        assert len(configs) == len(oss)
        self.configs = configs
        self.p_lats = nn.ModuleList()
        for config in self.configs:
            cat, idx = config.split("_")
            if cat == "i":
                self.p_lats.append(nn.Conv2d(incs[int(idx)], ouc, 1, stride=1, padding=0))
            elif cat == 'p':
                self.p_lats.append(nn.Conv2d(ouc, ouc, 1, stride=1, padding=0))
        
        biFPNLayers = []
        for _ in range(n_layers-1):
            biFPNLayers.append(biFPNLayer(ouc, oss, weighted_resize=weighted_resize))
        biFPNLayers.append(biFPNLayer(ouc, oss, fpn=fpn, weighted_resize=weighted_resize))
        self.biFPNLayers = nn.Sequential(*biFPNLayers)

    def forward(self, inputs): # low to high level, e.g., P3 -> P4 -> P5 ...
        ps = [] # high to low level, e.g., P7 -> P6 -> P5 ...
        for i, cfg in enumerate(self.configs):
            cat, idx = cfg.split("_")
            if cat == "i":
                ps.insert(0, self.p_lats[i](inputs[int(idx)]))               
            elif cat == 'p':
                ps.insert(0, self.p_lats[i](ps[0]))
        
        return self.biFPNLayers(ps)

class HSwish(nn.Module):
	def __init__(self):
		super(HSwish, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.gem = nn.AdaptiveAvgPool2d(1)
        self.p = p
        self.eps = eps
    def forward(self, x):
        return self.gem(x.clamp(min=self.eps).pow(self.p)).pow(1./self.p)

class HarmAttn(nn.Module):
    '''
    Harmonious Attention : Soft + Hard attention

    Soft(CxHxW) = Spatial(SpatialAttn, 1xHxW) x Channel(ChannelAttn, Cx1x1)
        Channel : GAP -> Conv 1x1xCx(C/r) -> Conv 1x1x(C/r)xC
        Spatial : channel-wise AP -> Conv 3x3 s2 -> Upx2 -> Conv 1x1

    Hard(HardAttn) (tx, ty) = entry of affine transform for feature map
        affine transform = 
            [[a, b, tx],
             [c, d, ty]]
        [tx, ty] = GAP -> fc -> tanh

    HarmAttn =
        theta -> STN
        soft -> STN -> Bilinear Interpolate (->  sum with last harmattn) -> block
    Args:
        block (nn.Module): module shared in all parallel branch
        n_stream (int): number of parallel branch
        channels (list): channels of module
            [inc, c1, c2, c3, ..., ouc(feat_dim)]
        shape: predefined shape of intermediate feature map
    '''
    def __init__(self, 
        block, 
        n_stream, 
        channels, 
        shape=(24, 28)):
        super(HarmAttn, self).__init__()
        self.n_stream = n_stream
        self.init_scale_factors()
        self.local = nn.ModuleList()
        self.shape = shape

        in_channel = channels[0]
        for out_channel in channels[1:-1]:
            self.local.append(block(in_channel, out_channel, 2))
            in_channel = out_channel

        feat_dim = channels[-1]
        self.local_fc = nn.Sequential(
            nn.Linear(out_channel * 4, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        self.pooling = GeM()

    def forward(self, stages):
        x_hatts = [None for _ in range(self.n_stream)]
        for stage_i, (x, theta) in enumerate(stages):
            for stream in range(self.n_stream):
                x_hatt = self.stn(
                    x, self.transform_theta(theta[:, stream, :], stream)
                )
                x_hatt = F.interpolate(
                    x_hatt, 
                    size=(self.shape[0]//(2**stage_i), self.shape[1]//(2**stage_i)),
                    mode='bilinear',
                    align_corners=True
                )
                if stage_i > 1:
                    x_hatt += x_hatts[stream]
                x_hatts[stream] = self.local[stage_i](x_hatt)
        
        for stream in range(self.n_stream):
            x_hatts[stream] = self.pooling(x_hatts[stream]).view(x_hatts[stream].size(0), -1)
        
        x_local = torch.cat(x_hatts, 1)
        x_local = self.local_fc(x_local)
        return x_local

    def init_scale_factors(self):
        # initialize scale factors (s_w, s_h) for four regions
        self.scale_factors = []
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )
        self.scale_factors.append(
            torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float)
        )

    def stn(self, x, theta):
        """Performs spatial transform
        
        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transforms theta to include (s_w, s_h), resulting in (batch, 2, 3)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:, :, :2] = scale_factors
        theta[:, :, -1] = theta_i
        if theta_i.is_cuda: theta = theta.cuda()
        return theta

class FPN(nn.Module):
    '''
    Args:
        config (list): 
            list of input feature type to be extracted by 
            lateral convolutional layer.
            For example, ["i_0", "i_1", "i_2", "i_2", "p_3"]
                where A_B,
                A: either "i" or "p", "i" means the feature is from input and "p" is from 
                    lateral convolutional  layer.
                B: the index of inputs or feature of lateral convolutional layer.
            In this example, 
            index 0 of input (i) -> lateral conv -> index 0 of lateral feature (p)
            index 1 of input (i) -> lateral conv -> index 1 of lateral feature (p)
            index 2 of input (i) -> lateral conv -> index 2 of lateral feature (p)
            index 2 of input (i) -> lateral conv -> index 3 of lateral feature (p)
            index 3 of lateral feature (p) -> lateral conv -> index 4 of lateral feature (p)

        incs (list):
            list of input channels for feature in config with "i" prefixed.

        oss (list):
            list of output stride for feature in config.

        oucs (list): 
            list of output channel for lateral convolutional layer that extracts feature from inputs

    '''
    def __init__(self, configs, incs, oss, oucs):
        super(FPN, self).__init__()
        assert len(configs) == len(oss)
        self.configs = configs
        self.p_lats = nn.ModuleList()
        self.p_ups = nn.ModuleList()
        for i, config in enumerate(self.configs):
            cat, idx = config.split("_")
            if cat == "i":
                self.p_lats.append(nn.Conv2d(incs[int(idx)], oucs[int(idx)], 1, stride=1, padding=0))
            elif cat == 'p':
                self.p_lats.append(nn.Conv2d(oucs[-1], oucs[-1], 1, stride=1, padding=0))
            if i > 0:
                if oss[i-1] * 2 == oss[i]:
                    self.p_ups.append(nn.Upsample(scale_factor=2))
                else:
                    self.p_ups.append(Identity())
        

    def forward(self, inputs): # low to high level, e.g., P3 -> P4 -> P5 ...
        ps = [] # low to high level, e.g., P3 -> P4 -> P5 ...
        for i, cfg in enumerate(self.configs):
            cat, idx = cfg.split("_")
            if cat == "i":
                x = self.p_lats[i](inputs[int(idx)])
            elif cat == 'p':
                x = self.p_lats[i](ps[i-1])
            ps.append(x)  # low to high level, e.g., P3 -> P4 -> P5 ...
        
        # high to low, e.g., P7 -> P6 -> P5 ...
        for i in range(len(ps)-2, -1, -1):
            ps[i] += self.p_ups[i](ps[i+1])       

        # low to high level, e.g., P3 -> P4 -> P5 ...      
        return ps
        
if __name__ == '__main__':
    # a = torch.rand([1,2,16,16])
    # b = torch.rand([1,3,8,8])
    # c = torch.rand([1,4,4,4])
    # model = biFPN([2,3,4], 5, num_layers=4)
    # model([a,b,c])

    a = torch.rand([1,4,16,16])
    b = torch.rand([1,4,8,8])
    c = torch.rand([1,4,4,4])
    d = torch.rand([1,4,4,4])

    # model = biFPNLayer(4, [1,2,4,4], fpn=True)
    # out = model([d,c,b,a])
    # model = biFPN(["i_0", "i_1", "i_2", "i_3"], [4,4,4,4], [1,2,4,4], 2, fpn=True, weighted_resize=True)
    # out = model([a,b,c,d])
    # print(out.shape)
    model = FPN(["i_0", "i_1", "i_2", "i_3"], [4,4,4,4], [1,2,4,4], 2)
    out = model([a,b,c,d])
    print(out.shape)

