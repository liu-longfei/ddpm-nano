#!/usr/bin/env python3
"""Train a tiny DDPM to memorize exactly one image."""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from ddpm import DDPM


def load_single_image(path, image_size=128):
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    # DDPM uses data in [-1, 1].
    x = to_tensor(image) * 2.0 - 1.0
    return x.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/target.png")
    parser.add_argument("--output", default="ddpm.pt")
    parser.add_argument("--steps", type=int, default=3000, help="training steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128, help="training resolution; must be divisible by 4")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.image_size <= 0 or args.image_size % 4:
        raise SystemExit("--image-size must be divisible by 4 because the U-Net downsamples twice")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    image = load_single_image(args.image, args.image_size).to(device)
    ddpm = DDPM(timesteps=1000).to(device)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        batch = image.repeat(args.batch_size, 1, 1, 1)
        loss = ddpm(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step:04d}/{args.steps:04d}  loss {loss.item():.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": ddpm.state_dict(),
            "image_size": args.image_size,
            "timesteps": ddpm.timesteps,
        },
        args.output,
    )
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
