import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _safe_group_norm(num_channels: int, groups: int, eps: float = 1e-6) -> nn.GroupNorm:
    """
    Pick the largest group count ≤ groups that divides num_channels.
    Falls back to 1 (instance-norm-like) if needed.
    """
    g = min(groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=num_channels, eps=eps)

# --------------------------
#   Small UNet primitives
# --------------------------

def conv3x3(in_ch: int, out_ch: int, bias: bool = True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)

class ResBlock(nn.Module):
    """
    Residual block: GN -> SiLU -> Conv -> GN -> SiLU -> Conv, with skip
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.norm1  = _safe_group_norm(in_ch, groups)   # CHANGED
        self.act1   = nn.SiLU(inplace=True)
        self.conv1  = conv3x3(in_ch, out_ch)
        self.norm2  = _safe_group_norm(out_ch, groups)  # CHANGED
        self.act2   = nn.SiLU(inplace=True)
        self.conv2  = conv3x3(out_ch, out_ch)
        self.skip   = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

        # Kaiming init
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(self.skip, nn.Conv2d):
            nn.init.kaiming_normal_(self.skip.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
            if self.skip.bias is not None: nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    """Downsample by 2 with a strided conv (safer than pooling for geofields)."""
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.op.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
        if self.op.bias is not None: nn.init.zeros_(self.op.bias)
    def forward(self, x): return self.op(x)

class Up(nn.Module):
    """Nearest-neighbor upsample + 3x3 conv to reduce checkerboard artifacts."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)
        nn.init.kaiming_normal_(self.conv.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
        if self.conv.bias is not None: nn.init.zeros_(self.conv.bias)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# --------------------------
#     ForecastModel
# --------------------------

class ForecastModel(nn.Module):
    """
    Simple, strong UNet for next-timestep forecasting on MERRA-2 grids.

    - Accepts arbitrary HxW; internally pads to a multiple of 8 and crops back.
    - Output size == input size.
    - Uses GroupNorm+SiLU for stability on standardized (z-scored) inputs.

    Args
    ----
    in_channels:   C_in  (len(inputs) in config/train.py)
    out_channels:  C_out (len(targets))
    img_resolution: (H, W) — not enforced rigidly; used for info/logging
    dim:           base width (channels) of the UNet (default 256)
    depth:         overall depth hint; we use 3 downs/ups (x8) for safety on 361x576
    heads, patch_size, window_size, flash: kept for API parity; not used here
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_resolution: Tuple[int, int] = (361, 576),
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        patch_size: Tuple[int, int] = (1, 2),
        window_size: Tuple[int, int] = (4, 8),
        flash: bool = True,
        groups: int = 8,                      # NEW
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_resolution = img_resolution
        self.gn_groups = groups               # NEW

        # Channel plan: 3 downs (x8) is a sweet spot for 361x576 with padding
        c1 = max(32, dim // 4)
        c2 = max(64, dim // 2)
        c3 = dim
        c4 = dim * 2  # bottleneck

        # Encoder
        self.enc1 = nn.Sequential(ResBlock(in_channels, c1, groups=self.gn_groups),
                                ResBlock(c1, c1, groups=self.gn_groups))
        self.down1 = Down(c1)

        self.enc2 = nn.Sequential(ResBlock(c1, c2, groups=self.gn_groups),
                                ResBlock(c2, c2, groups=self.gn_groups))
        self.down2 = Down(c2)

        self.enc3 = nn.Sequential(ResBlock(c2, c3, groups=self.gn_groups),
                                ResBlock(c3, c3, groups=self.gn_groups))
        self.down3 = Down(c3)

        # Bottleneck
        self.mid = nn.Sequential(ResBlock(c3, c4, groups=self.gn_groups),
                                ResBlock(c4, c4, groups=self.gn_groups))

        # Decoder
        self.up3 = Up(c4, c3)
        self.dec3 = nn.Sequential(ResBlock(c3 + c3, c3, groups=self.gn_groups),
                                ResBlock(c3, c3, groups=self.gn_groups))

        self.up2 = Up(c3, c2)
        self.dec2 = nn.Sequential(ResBlock(c2 + c2, c2, groups=self.gn_groups),
                                ResBlock(c2, c2, groups=self.gn_groups))

        self.up1 = Up(c2, c1)
        self.dec1 = nn.Sequential(ResBlock(c1 + c1, c1, groups=self.gn_groups),
                                ResBlock(c1, c1, groups=self.gn_groups))

        # Head
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, m: int = 8):
        """
        Pad H and W to be multiples of m. Returns (x_pad, (pad_left, pad_right, pad_top, pad_bottom)).
        """
        _, _, H, W = x.shape
        Ht = (H + m - 1) // m * m
        Wt = (W + m - 1) // m * m
        pad_h = Ht - H
        pad_w = Wt - W
        # F.pad pads as (left, right, top, bottom)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x_pad, (0, pad_w, 0, pad_h)

    @staticmethod
    def _crop_to(x: torch.Tensor, H: int, W: int):
        return x[..., :H, :W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]  (standardized)
        y: [B, C_out, H, W] (standardized)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Safe pad to multiple of 8
        x, pads = self._pad_to_multiple(x, m=8)

        # Encoder
        e1 = self.enc1(x)        # [B, c1, H, W]
        d1 = self.down1(e1)      # [B, c1, H/2, W/2]

        e2 = self.enc2(d1)       # [B, c2, H/2, W/2]
        d2 = self.down2(e2)      # [B, c2, H/4, W/4]

        e3 = self.enc3(d2)       # [B, c3, H/4, W/4]
        d3 = self.down3(e3)      # [B, c3, H/8, W/8]

        # Bottleneck
        m  = self.mid(d3)        # [B, c4, H/8, W/8]

        # Decoder
        u3 = self.up3(m)         # -> [B, c3, H/4, W/4]
        # align and concat skip (handle potential 1px mismatches safely)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="nearest")
        x3 = torch.cat([u3, e3], dim=1)
        x3 = self.dec3(x3)       # [B, c3, H/4, W/4]

        u2 = self.up2(x3)        # -> [B, c2, H/2, W/2]
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="nearest")
        x2 = torch.cat([u2, e2], dim=1)
        x2 = self.dec2(x2)       # [B, c2, H/2, W/2]

        u1 = self.up1(x2)        # -> [B, c1, H, W]
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="nearest")
        x1 = torch.cat([u1, e1], dim=1)
        x1 = self.dec1(x1)       # [B, c1, H, W]

        y  = self.head(x1)       # [B, C_out, H, W]

        # Crop back to original HxW (undo padding)
        y = self._crop_to(y, H, W)
        return y
