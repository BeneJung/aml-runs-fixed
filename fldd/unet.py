import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class BlockOutputHead(nn.Module):
    """Output head that produces logits over block states.

    For block_size=1: (B, 1, H, W) per-pixel logit (backward compatible)
    For block_size=2, orientation='horizontal' (default): (B, 4, H, W//2)
        4-way categorical per 1x2 horizontal block
    For block_size=2, orientation='vertical': (B, 4, H//2, W)
        4-way categorical per 2x1 vertical block (B11 control)
    For block_size=4: (B, 16, H//2, W//2) 16-way categorical per 2x2 block
    """

    def __init__(self, in_channels, block_size, orientation="horizontal"):
        super().__init__()
        self.block_size = block_size
        self.orientation = orientation
        n_states = 2 ** block_size

        if block_size == 1:
            self.head = nn.Conv2d(in_channels, 1, 1)
        elif block_size == 2:
            if orientation == "vertical":
                # 2x1 vertical blocks
                self.head = nn.Conv2d(
                    in_channels, n_states, kernel_size=(2, 1), stride=(2, 1),
                )
            else:
                self.head = nn.Conv2d(
                    in_channels, n_states, kernel_size=(1, 2), stride=(1, 2),
                )
        elif block_size == 4:
            self.head = nn.Conv2d(in_channels, n_states, kernel_size=2, stride=2)
        else:
            raise ValueError(f"unsupported block_size={block_size}")

    def forward(self, features):
        return self.head(features)


class UNet(nn.Module):
    """Small U-Net for 28x28 binary images.

    Takes noisy image z_t (1 channel) and timestep t.
    Output depends on block_size:
      block_size=1: (B, 1, 28, 28) per-pixel logits
      block_size=2: (B, 4, 28, 14) logits over 1x2 block states
      block_size=4: (B, 16, 14, 14) logits over 2x2 block states
    """

    def __init__(self, channels=(32, 64, 128), t_dim=64, block_size=1,
                 orientation="horizontal"):
        super().__init__()
        self.t_dim = t_dim
        self.block_size = block_size
        self.orientation = orientation
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.SiLU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # input conv
        self.in_conv = nn.Conv2d(1, channels[0], 3, padding=1)

        # down path: each level does ResBlock then downsample
        # skip connections are saved BEFORE downsampling.
        # NOTE (hygiene): historical versions of this code initialised
        # skip_channels as [channels[0]] to track the in_conv output,
        # but the up loop only pops len(channels)-1 = 2 times, leaving
        # the in_conv entry unused. The model architecture and trained
        # weights are unchanged; we just drop the dead bookkeeping so
        # the up-path construction reads cleanly.
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = []
        in_ch = channels[0]
        for ch in channels[1:]:
            self.down_blocks.append(ResBlock(in_ch, ch, t_dim))
            self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            skip_channels.append(ch)
            in_ch = ch

        # middle
        self.mid = ResBlock(channels[-1], channels[-1], t_dim)

        # up path: upsample, concat skip, ResBlock
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in enumerate(reversed(channels[:-1])):
            skip_ch = skip_channels.pop()
            self.upsamples.append(nn.ConvTranspose2d(in_ch, ch, 4, stride=2, padding=1))
            self.up_blocks.append(ResBlock(ch + skip_ch, ch, t_dim))
            in_ch = ch

        # output: norm + block head
        self.out_norm = nn.GroupNorm(min(8, channels[0]), channels[0])
        self.out_head = BlockOutputHead(channels[0], block_size, orientation=orientation)

    def forward(self, z_t, t):
        """
        Args:
            z_t: (B, 1, 28, 28) noisy binary image
            t: (B,) integer timestep indices

        Returns:
            logits: block-level logits (shape depends on block_size)
        """
        t_emb = self.time_mlp(t)
        h = self.in_conv(z_t)

        # down. We DO NOT skip from in_conv (see the construction note);
        # only the post-ResBlock features are passed up.
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        # middle
        h = self.mid(h, t_emb)

        # up
        for block, up in zip(self.up_blocks, self.upsamples):
            h = up(h)
            skip = skips.pop()
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:])
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        h = F.silu(self.out_norm(h))
        return self.out_head(h)
