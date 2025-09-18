import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn import Conv3d


def build_onehot_kernels_3d(k: int, device=None, dtype=None):
    """
    返回形如 (k^3, 1, k, k, k) 的卷积核，每个核在一个体素处为1，其余为0。
    用它对单通道体数据做 conv3d + padding=k//2，可得到每个中心位置的所有邻域体素。
    对多通道向量场，用 groups=C 的 conv3d 同时对各通道抽取邻域。
    """
    K = torch.zeros((k**3, 1, k, k, k), device=device, dtype=dtype)
    idx = 0
    for z in range(k):
        for y in range(k):
            for x in range(k):
                K[idx, 0, z, y, x] = 1.0
                idx += 1
    return K  # (k^3, 1, k, k, k)


# ---------- 单步 3D 聚合器（凸组合） ----------
class _StepAggregator3D(nn.Module):
    def __init__(self, in_ch, k=3, hidden=32, pixelwise_gate=True, training = False):
        super().__init__()
        assert k % 2 == 1, "k must be odd"
        self.k = k
        self.pad = k // 2
        self.pixelwise_gate = pixelwise_gate

        # 条件特征 = 输入特征 + 当前 flow(3通道)
        self.backbone = nn.Sequential(
            nn.Conv3d(in_ch + 3, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 1, padding=0), torch.nn.BatchNorm3d(hidden), nn.ReLU(inplace=True),
        )
        self.kernel_head = nn.Conv3d(hidden, k*k*k, 1)  # (B,k^3,D,H,W)

        if pixelwise_gate:
            self.gate = nn.Sequential(nn.Conv3d(hidden, 1, 1), nn.Softplus())  # >=0
        else:
            self.scale = nn.Parameter(torch.tensor(0.0))  # 全局 softplus 步幅

        # one-hot 3D 卷积核（在 forward 里按 dtype/device 缓存/注册）
        self.register_buffer("_kernels", torch.empty(0))  # 延迟构造

    def _ensure_kernels(self, x):
        if self._kernels.numel() == 0 or self._kernels.device != x.device or self._kernels.dtype != x.dtype:
            self._kernels = build_onehot_kernels_3d(self.k, x.device, x.dtype)  # (k^3,1,k,k,k)

    def forward(self, x_feat, flow):
        B, _, D, H, W = x_feat.shape
        K3 = self.k ** 3
        self._ensure_kernels(x_feat)

        feat = self.backbone(torch.cat([x_feat, flow], dim=1))  # (B,hidden,D,H,W)
        logits = self.kernel_head(feat)  # (B,K3,D,H,W)
        W = F.softmax(logits.view(B, K3, -1), dim=1).view(B, K3, D, H, W)

        # —— 逐通道抽邻域并加权 —— 显存友好版
        delta_ch = []
        for c in range(3):
            # 对 flow 的第 c 通道做 conv3d，得到 k^3 邻域堆栈 (B,K3,D,H,W)
            neigh_c = F.conv3d(flow[:, c:c + 1], self._kernels, padding=self.pad)  # groups=1
            # 与 W 做逐核加权和 -> (B,1,D,H,W)
            d_c = (W * neigh_c).sum(dim=1, keepdim=True)
            delta_ch.append(d_c)

        delta = torch.cat(delta_ch, dim=1)  # (B,3,D,H,W)

        if self.pixelwise_gate:
            alpha = self.gate(feat)  # (B,1,D,H,W) >= 0
            delta = delta * (1.0 + alpha)
        else:
            delta = delta * (1.0 + F.softplus(self.scale))

        return delta  # 训练外再返回 W，训练中可直接返回 None


# ---------- 迭代式 3D 形变网络 ----------
class IterNeighborhoodDeform3D(nn.Module):
    """
    flow_0 由 mapper 给出（或全零），迭代 T 步：
      flow_{t} = flow_{t-1} + step * Aggregator3D(x, flow_{t-1})
    """
    def __init__(self, in_ch=2, k=3, hidden=32, steps=5,
                 map_to_vec=True, init_zero=False,
                 pixelwise_gate=True, learn_step=True):
        super().__init__()
        self.steps = steps
        self.init_zero = init_zero

        self.map_to_vec = map_to_vec and (not init_zero)
        if self.map_to_vec:
            self.mapper = nn.Sequential(
                nn.Conv3d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(hidden, hidden, 1, padding=0), torch.nn.BatchNorm3d(hidden), nn.ReLU(inplace=True),
                nn.Conv3d(hidden, 3, 1)  # -> (B,3,D,H,W)
            )

        self.stepper = _StepAggregator3D(in_ch=in_ch, k=k, hidden=hidden,
                                         pixelwise_gate=pixelwise_gate)

        if learn_step:
            self.log_step = nn.Parameter(torch.tensor(0.0))
            self.learn_step = True
        else:
            self.register_buffer('fixed_step', torch.tensor(1.0))
            self.learn_step = False

    def _init_flow(self, x):
        B, _, D, H, W = x.shape
        if self.init_zero or (not self.map_to_vec):
            return torch.zeros(B, 3, D, H, W, device=x.device, dtype=x.dtype)
        return self.mapper(x)

    def forward(self, x, return_all=False):
        """
        x: (B, C, D, H, W)
        return:
          flow_T: (B,3,D,H,W)
          extras: {"flows":list, "weights":list, "step":tensor} （可选）
        """
        flow = self._init_flow(x)
        flows, weights = [flow], []

        for _ in range(self.steps):
            delta = self.stepper(x, flow)
            step = F.softplus(self.log_step) if self.learn_step else self.fixed_step
            flow = flow + step * delta
            flows.append(flow)

            return flow, {}