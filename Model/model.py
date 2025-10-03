import math

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
    def __init__(self, dim, kernel_size=7, steps=10, fire_rate=1, n_channels=16, hidden_size=64, flow_param='cartesian', max_disp = 8.0):
        # def __init__(self, kernel_size = 7, steps = 50, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 90, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate= 0.25, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 0.5, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 0.75, n_channels = 16, hidden_size = 64):
        # def __init__(self, kernel_size = 7, steps = 10, fire_rate = 1.0, n_channels = 16, hidden_size = 64):

        super().__init__()
        # -- Set variable that defines number of feature channels for NCAs output after forward pass -- #
        self.out_feats = n_channels  # Set this dynamically
        self.dim = dim
        self.flow_param = flow_param
        self.max_disp = max_disp
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

        dir_dim = 3 if self.dim == 3 else 2
        if self.flow_param == 'spherical':
            out_c = dir_dim + 1  # [direction, rho]
        else:  # 'cartesian' or 'svf'
            out_c = dir_dim  # 笛卡尔 (u,v,w) 或 速度 (vx,vy,vz)
        self.flow = Conv3d(self.out_feats, out_c, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def _grid_identity(self, shape, device):
        # shape = (B, C, D, H, W); 返回标准化坐标网格 [-1,1]
        B, _, D, H, W = shape
        zs = torch.linspace(-1, 1, D, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        z, y, x = torch.meshgrid(zs, ys, xs, indexing='ij')
        grid = torch.stack((x, y, z), dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B,3,D,H,W)
        return grid

    def _warp(self, field, disp):  # field: (B,C,D,H,W); disp: 像素位移 → 需归一化到[-1,1]
        B, _, D, H, W = field.shape
        norm = torch.tensor([W - 1, H - 1, D - 1], device=field.device).view(1, 3, 1, 1, 1)
        grid = self._grid_identity(field.shape, field.device) + 2.0 * disp / norm
        grid = grid.permute(0, 2, 3, 4, 1)  # (B,D,H,W,3)
        return F.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def _compose(self, d1, d2):
        # compose displacements: ϕ = d1 ∘ d2 = d2 + warp(d1, d2)
        return d2 + self._warp(d1, d2)

    def _exp_velocity(self, v, n=7):
        # scaling-and-squaring
        v = v / (2 ** n)
        disp = v
        for _ in range(n):
            disp = self._compose(disp, disp)
        return disp
    def _decode_flow_vec(self, q, max_disp=None):
        # q: (B, C, D, H, W)  ->  C = (dir_dim + 1)
        dir_dim = 3 if self.dim == 3 else 2
        if self.flow_param == 'spherical':
            v = q[:, :dir_dim]
            rho_raw = q[:, dir_dim:dir_dim + 1]
            v = v / (v.norm(dim=1, keepdim=True) + 1e-6)
            rho = (max_disp * torch.sigmoid(rho_raw)) if (max_disp is not None) else F.softplus(rho_raw)
            flow = rho * v
        elif self.flow_param in ['cartesian', 'svf']:
            flow = q
            if max_disp is not None:  # 限幅，防梯度爆
                flow = max_disp * torch.tanh(flow)
        return flow

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
        B, C, D, H, W = x.shape  # x: (B, 2, D, H, W) moving/fixed
        device = x.device
        x_full = torch.zeros((B, self.n_channels, D, H, W), dtype=torch.float32, device=device)
        x_full[:, 0:2, ...] = x
        x_downscaled = self.avg_pool(x_full)

        for step in range(self.steps):
            x_downscaled = self.update(x_downscaled, step)

        x_feat = self.up(x_downscaled)

        # 尺寸对齐（保持你的原逻辑）
        if x_full.size() != x_feat.size():
            padC = x_full.size(1) - x_feat.size(1)
            if padC > 0:
                x_feat = torch.cat([x_feat, torch.zeros(B, padC, x_feat.size(2), x_feat.size(3), x_feat.size(4), device=device)], dim=1)
            padZ = x_full.size(2) - x_feat.size(2)
            if padZ > 0:
                x_feat = torch.cat([x_feat, torch.zeros(B, x_feat.size(1), padZ, x_feat.size(3), x_feat.size(4), device=device)], dim=2)
            padY = x_full.size(3) - x_feat.size(3)
            if padY > 0:
                x_feat = torch.cat([x_feat, torch.zeros(B, x_feat.size(1), x_feat.size(2), padY, x_feat.size(4), device=device)], dim=3)
            padX = x_full.size(4) - x_feat.size(4)
            if padX > 0:
                x_feat = torch.cat([x_feat, torch.zeros(B, x_feat.size(1), x_feat.size(2), x_feat.size(3), padX, device=device)], dim=4)

        q = self.flow(x_feat)          # q = [θ, ρ] 或 [θ, φ, ρ] 或 [u,v,(w)]
        if self.flow_param == 'svf':
            v = self._decode_flow_vec(q, max_disp=self.max_disp)  # 这里的max_disp可较大，后续指数映射会抑制过大位移
            pos_flow = self._exp_velocity(v, n=7)
        else:
            pos_flow = self._decode_flow_vec(q, max_disp=self.max_disp)

        return pos_flow




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
