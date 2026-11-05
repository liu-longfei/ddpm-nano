"""Minimal class-conditional DDPM on top of the tiny DDPM implementation."""
from torch import nn

import torch
from ddpm import DDPM, TinyUNet, timestep_embedding


class ConditionalTinyUNet(TinyUNet):
    """TinyUNet where a class label is added to the timestep embedding."""
    def __init__(self, base=32, time_dim=64, num_classes=10):
        super().__init__(base=base, time_dim=time_dim)
        self.label_embedding = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features))
        t_emb = t_emb + self.label_embedding(y)
        x1 = self.down1(self.stem(x), t_emb)
        x2 = self.down2(self.downsample1(x1), t_emb)
        x3 = self.down3(self.downsample2(x2), t_emb)
        x = self.mid(x3, t_emb)
        x = self.up3(torch.cat([self.upsample3(x), x2], dim=1), t_emb)
        x = self.up2(torch.cat([self.upsample2(x), x1], dim=1), t_emb)
        return self.out(x)


class ConditionalDDPM(DDPM):
    """DDPM whose noise prediction is conditioned on a class label."""
    def forward(self, x0, y):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        xt = sqrt_ac * x0 + sqrt_1mac * noise
        pred_noise = self.model(xt, t, y)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, shape, y, steps=50, eta=0.0, device=None):
        device = device or next(self.model.parameters()).device
        self.model.eval()
        x = torch.randn(shape, device=device)
        seq = torch.linspace(0, self.timesteps - 1, steps, device=device).round().long().flip(0)

        for i, t in enumerate(seq):
            t_batch = t.repeat(shape[0])
            pred_noise = self.model(x, t_batch, y)
            x0 = (x - pred_noise * self.sqrt_one_minus_alphas_cumprod[t]) / self.sqrt_alphas_cumprod[t]
            x0 = x0.clamp(-1, 1)

            if i + 1 < len(seq):
                ab_t = self.sqrt_alphas_cumprod[t] ** 2
                ab_prev = self.sqrt_alphas_cumprod[seq[i + 1]] ** 2
                sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
                direction = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * pred_noise
                x = ab_prev.sqrt() * x0 + direction
                if eta > 0 and sigma > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0
        return x
