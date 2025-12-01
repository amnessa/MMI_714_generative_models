import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange


# --- helper modules (adapted from lucidrains/denoising-diffusion-pytorch style) ---


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    """Standard 1D sinusoidal time embedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b,) scalar timesteps or continuous time
        device = x.device
        half_dim = self.dim // 2
        emb_factor = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            # pad to odd dim if needed
            emb = F.pad(emb, (0, 1))
        return emb


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None):
        super().__init__()

        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out * 2),
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )

        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1[0](x)
        h = self.block1[1](h)
        h = self.block1[2](h)

        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.op = nn.Conv2d(dim, dim_out, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, dim_out, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Attention(nn.Module):
    """Simple multi-head self-attention block for 2D feature maps."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.GroupNorm(8, dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)

        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=self.heads),
            qkv,
        )

        q = q * self.scale
        attn = torch.softmax(torch.einsum("b h i d, b h j d -> b h i j", q, k), dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return out + x


# --- Conditional U-Net for depth completion ---


class ConditionalUNetDepth(nn.Module):
    """
    Conditional U-Net operating in pixel space for depth completion.

    Inputs
    ------
    x_noisy_depth : (B, 1, H, W)
        Noisy depth map at timestep t.
    cond_rgbd    : (B, 4, H, W)
        Conditioning tensor: 3-channel RGB + 1-channel raw / incomplete depth.
    t            : (B,) or (B, 1)
        Diffusion timestep or continuous time value.

    Output
    ------
    pred_noise   : (B, 1, H, W)
        Predicted noise residual added to the clean ground truth depth.
    """

    def __init__(
        self,
        *,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        use_attention_resolutions=(16,),
        time_emb_dim: int = 256,
        cond_channels: int = 4,
    ):
        super().__init__()

        self.in_channels_x = 1
        # e.g. 3xRGB + 1x raw / incomplete depth [+ optional guidance maps]
        self.in_channels_cond = cond_channels
        self.out_channels = 1      # noise on depth

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        init_dim = base_channels
        self.x_in = nn.Conv2d(self.in_channels_x + self.in_channels_cond, init_dim, 3, padding=1)

        # encoder
        self.downs = nn.ModuleList([])
        self.skip_channels = []  # channels for each skip connection

        in_ch = init_dim
        curr_res = None  # keep abstract for now (no resolution-based attention)

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResnetBlock(in_ch, out_ch, time_emb_dim=time_emb_dim)
                attn = Attention(out_ch) if (curr_res in use_attention_resolutions) else nn.Identity()
                self.downs.append(nn.ModuleDict({"block": block, "attn": attn}))
                in_ch = out_ch
                self.skip_channels.append(in_ch)

            # add downsample between levels, except at the last one
            if i != len(channel_mults) - 1:
                self.downs.append(nn.ModuleDict({"downsample": Downsample(in_ch, in_ch)}))

        # bottleneck
        self.mid_block1 = ResnetBlock(in_ch, in_ch, time_emb_dim=time_emb_dim)
        self.mid_attn = Attention(in_ch)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, time_emb_dim=time_emb_dim)

        # decoder
        self.ups = nn.ModuleList([])

        # we will traverse skip connections in reverse order
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                # take the last recorded skip channel for concatenation
                skip_ch = self.skip_channels.pop()
                self.ups.append(
                    nn.ModuleDict(
                        {
                            "block": ResnetBlock(in_ch + skip_ch, out_ch, time_emb_dim=time_emb_dim),
                            "attn": Attention(out_ch) if (curr_res in use_attention_resolutions) else nn.Identity(),
                        }
                    )
                )
                in_ch = out_ch

            if i != 0:
                self.ups.append(nn.ModuleDict({"upsample": Upsample(in_ch, in_ch)}))

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, self.out_channels, 3, padding=1)

    def forward(
        self,
        x_noisy_depth: torch.Tensor,
        cond_rgbd: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_noisy_depth : (B, 1, H, W)
        cond_rgbd    : (B, 4, H, W)
        t            : (B,) or (B, 1)

        Returns
        -------
        pred_noise   : (B, 1, H, W)
        """

        assert x_noisy_depth.shape[1] == self.in_channels_x, "x_noisy_depth must have 1 channel"
        assert cond_rgbd.shape[1] == self.in_channels_cond, f"cond_rgbd must have {self.in_channels_cond} channels"

        if t.dim() == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        assert t.dim() == 1, "t must be shape (B,) or (B, 1)"

        time_emb = self.time_mlp(t)

        x = torch.cat([x_noisy_depth, cond_rgbd], dim=1)
        x = self.x_in(x)

        hs = []

        # encoder path
        for module in self.downs:
            if isinstance(module, nn.ModuleDict) and ("block" in module):
                x = module["block"](x, time_emb=time_emb)
                x = module["attn"](x)
                hs.append(x)
            elif isinstance(module, nn.ModuleDict) and ("downsample" in module):
                x = module["downsample"](x)

        # bottleneck
        x = self.mid_block1(x, time_emb=time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=time_emb)

        # decoder path
        for module in self.ups:
            if isinstance(module, nn.ModuleDict) and ("block" in module):
                skip = hs.pop()
                x = torch.cat([x, skip], dim=1)
                x = module["block"](x, time_emb=time_emb)
                x = module["attn"](x)
            elif isinstance(module, nn.ModuleDict) and ("upsample" in module):
                x = module["upsample"](x)

        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    # quick sanity check
    model = ConditionalUNetDepth(cond_channels=5)
    b, h, w = 2, 128, 128
    x_noisy = torch.randn(b, 1, h, w)
    cond = torch.randn(b, 5, h, w)
    t = torch.randint(0, 1000, (b,), dtype=torch.long)

    with torch.no_grad():
        out = model(x_noisy, cond, t)
    print("output shape:", out.shape)
