#!/usr/bin/env python3
"""Generate the learned image from a trained tiny DDPM checkpoint."""
import argparse

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from ddpm import DDPM, TinyUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="ddpm.pt")
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--image-size", type=int, default=None, help="sampling resolution override; defaults to checkpoint image_size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    image_size = args.image_size if args.image_size is not None else checkpoint["image_size"]
    if image_size <= 0 or image_size % 4:
        raise SystemExit("--image-size must be divisible by 4 because the U-Net downsamples twice")

    model = TinyUNet()
    ddpm = DDPM(model=model, timesteps=checkpoint["timesteps"]).to(device)
    ddpm.load_state_dict(checkpoint["model"])
    ddpm.eval()

    torch.manual_seed(args.seed)
    with torch.no_grad():
        image = ddpm.ddim_sample((1, 3, image_size, image_size), steps=args.steps)
    image = (image.clamp(-1, 1) + 1.0) / 2.0
    to_pil_image(image[0].cpu()).save(args.output)
    print(f"saved image: {args.output}")


if __name__ == "__main__":
    main()
