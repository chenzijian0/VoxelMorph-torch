import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import Conv3d


class NCA(nn.Module):
    """
    A NCA architecture for segmentation.
    """

    # def __init__(self, kernel_size = 3, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 5, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 9, steps = 30, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    # def __init__(self, kernel_size = 7, steps = 5, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
    def __init__(self, dim, kernel_size=3, steps=10, fire_rate=1, n_channels=16, hidden_size=64):
        # def __init__(self, kernel_size = 7, steps = 50, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 90, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate= 0.25, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 0.75, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 1.0, n_channels = 16, hidden_size = 64):
        r"""
        Parameters:
            kernel_size: Kernel size of NCA -> Relevant for perceptive field -> perceptive field = (kernel_size-1)/2 * steps
            steps: Times the NCA model will be applied to the input
            fire_rate = Chance that a cell is active at given step
            n_channels = Channels of NCA -> In channels are equal to out channels
            hidden_size = Hidden size of NCA
        """
        super().__init__()
        # -- Set variable that defines number of feature channels for NCAs output after forward pass -- #
        self.out_feats = n_channels  # Set this dynamically
        self.dim = dim
        # Model components
        # self.fc0 = nn.Linear(n_channels * 2, hidden_size)
        # self.fc1 = nn.Linear(hidden_size, n_channels, bias=False)
        padding = int((kernel_size - 1) / 2)
        self.p0 = nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding,
                            padding_mode="reflect")
        self.p1 = nn.Conv3d(n_channels * 2, hidden_size, kernel_size=1, stride=1, padding=0)
        self.p2 = nn.Conv3d(hidden_size, n_channels, kernel_size=1, stride=1, padding=0)
        self.p = nn.ModuleList()

        self.mian = nn.ModuleList()

        for i in range(steps):
            self.mian.append(nn.Conv3d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=padding,
                            padding_mode="reflect"))
            self.p.append(self.conv_block(n_channels * 2, hidden_size, 1))
            self.p.append(self.conv_block(hidden_size, n_channels, 1))

        # self.bn = torch.nn.BatchNorm3d(hidden_size)

        # self.avg_pool = torch.nn.AvgPool3d(3, 2, 1)
        # self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = torch.nn.AvgPool3d(3, 3, 0)
        self.up = torch.nn.Upsample(scale_factor=3, mode='nearest')
        # self.avg_pool = torch.nn.AvgPool3d(5, 4, 2)
        # self.up = torch.nn.Upsample(scale_factor=4, mode='nearest')

        # Model settings
        self.fire_rate = fire_rate
        self.steps = steps
        self.n_channels = n_channels

        self.flow = Conv3d(self.out_feats, dim, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def conv_block(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer
    def perceive(self, x, ite):
        y = self.mian[ite](x)
        y = torch.cat((x, y), 1)
        return y

    def update(self, x_in, ite):
        # x = x_in.transpose(1,4)
        # dx = self.perceive(x)
        dx = self.perceive(x_in, ite)
        # dx = dx.transpose(1, 4)
        # dx = self.fc0(dx)
        # dx = self.p1(dx)
        # dx = dx.transpose(1, 4)
        # dx = self.bn(dx)
        # dx = dx.transpose(1, 4)
        # dx = F.relu(dx)
        # dx = dx.transpose(1, 4)
        # dx = self.p2(dx)
        dx = self.p[ite*2](dx)
        dx = self.p[ite*2+1](dx)

        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), dx.size(3), 1]) < self.fire_rate
        stochastic = stochastic.float().cuda()
        dx = dx * stochastic
        x = x_in + dx
        # x = x.transpose(1,4)

        return x

    def forward(self, x):
        r"""
        Forward pass.
        """
        # Prepare input
        x_full = torch.zeros((x.shape[0], self.n_channels, x.shape[2], x.shape[3], x.shape[4]),
                             dtype=torch.float32).cuda()
        x_full[:, 0:2, ...] = x
        x_downscaled = self.avg_pool(x_full)
        # x_downscaled = x_full

        for step in range(self.steps):
            x_downscaled = self.update(x_downscaled, step)

        x = self.up(x_downscaled)

        if x_full.size() != x.size():
            # -- Zero pad to original size -- #
            x_ = torch.zeros((x.shape[0], x_full.size(1) - x.size(1), x.shape[2], x.shape[3], x.shape[4]),
                             dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=1)
            x_ = torch.zeros((x.shape[0], x.shape[1], x_full.size(2) - x.size(2), x.shape[3], x.shape[4]),
                             dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=2)
            x_ = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x_full.size(3) - x.size(3), x.shape[4]),
                             dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=3)
            x_ = torch.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3], x_full.size(4) - x.size(4)),
                             dtype=torch.float32).cuda()
            x = torch.concat([x, x_], dim=4)

        flow_field = self.flow(x)
        # resize flow for integration
        pos_flow = flow_field
        return pos_flow
        # return x_downscaled




class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
