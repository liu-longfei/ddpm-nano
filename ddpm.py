"""A very small DDPM implementation with a tiny U-Net."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal timestep embedding. t: [B], output: [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """A compact U-Net for RGB images whose size is divisible by 4."""
    def __init__(self, base=32, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, time_dim),
        )
        self.stem = nn.Conv2d(3, base, 3, padding=1)

        self.down1 = ResBlock(base, base, time_dim)
        self.downsample1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.down2 = ResBlock(base * 2, base * 2, time_dim)
        self.downsample2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.down3 = ResBlock(base * 4, base * 4, time_dim)

        self.mid = ResBlock(base * 4, base * 4, time_dim)

        self.upsample3 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.up3 = ResBlock(base * 4, base * 2, time_dim)
        self.upsample2 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.up2 = ResBlock(base * 2, base, time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 3, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features))
        x1 = self.down1(self.stem(x), t_emb)
        x2 = self.down2(self.downsample1(x1), t_emb)
        x3 = self.down3(self.downsample2(x2), t_emb)
        x = self.mid(x3, t_emb)
        x = self.up3(torch.cat([self.upsample3(x), x2], dim=1), t_emb)
        x = self.up2(torch.cat([self.upsample2(x), x1], dim=1), t_emb)
        return self.out(x)


class DDPM(nn.Module):
    def __init__(self, model=None, timesteps=1000):
        super().__init__()
        self.model = model or TinyUNet()
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt().float())

    def forward(self, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        xt = sqrt_ac * x0 + sqrt_1mac * noise
        pred_noise = self.model(xt, t)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, shape, steps=50, eta=0.0, device=None):
        """Fast deterministic DDIM sampling; steps=50 is usually enough."""
        device = device or next(self.model.parameters()).device
        self.model.eval()
        x = torch.randn(shape, device=device)
        # Reverse through evenly spaced timesteps, e.g. 999 -> 0.
        seq = torch.linspace(0, self.timesteps - 1, steps, device=device).round().long().flip(0)

        for i, t in enumerate(seq):
            t_batch = t.repeat(shape[0])
            pred_noise = self.model(x, t_batch)
            ab_t = self.sqrt_alphas_cumprod[t] ** 2
            x0 = (x - pred_noise * self.sqrt_one_minus_alphas_cumprod[t]) / self.sqrt_alphas_cumprod[t]
            x0 = x0.clamp(-1, 1)

            if i + 1 < len(seq):
                ab_prev = self.sqrt_alphas_cumprod[seq[i + 1]] ** 2
                sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
                direction = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
                x = ab_prev.sqrt() * x0 + direction
                if eta > 0 and sigma > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                # At t=0, DDIM with eta=0 returns x0 directly.
                x = x0
        return x
