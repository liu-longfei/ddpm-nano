#!/usr/bin/env python3
"""Train a tiny class-conditional DDPM.

With the default setup it trains class 0 from data/target.png.  Put several
images into data/conditional/ to train one class per image (sorted by name).
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from cond_ddpm import ConditionalDDPM, ConditionalTinyUNet

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path, image_size=128):
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    return to_tensor(image) * 2.0 - 1.0


def load_condition_data(data_dir, fallback_image, image_size):
    data_dir = Path(data_dir)
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) if data_dir.exists() else []
    if not paths:
        paths = [Path(fallback_image)]
    images = [load_image(path, image_size) for path in paths]
    return images, [path.name for path in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/conditional", help="one image per class")
    parser.add_argument("--image", default="data/target.png", help="single-image fallback")
    parser.add_argument("--output", default="conditional_ddpm.pt")
    parser.add_argument("--steps", type=int, default=10000, help="training steps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128, help="training resolution; must be divisible by 4")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.image_size <= 0 or args.image_size % 4:
        raise SystemExit("--image-size must be divisible by 4 because the U-Net downsamples twice")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images, class_names = load_condition_data(args.data_dir, args.image, args.image_size)
    images = [image.to(device) for image in images]
    num_classes = len(images)
    print(f"classes: {num_classes} ({', '.join(class_names)})")

    model = ConditionalTinyUNet(num_classes=num_classes)
    ddpm = ConditionalDDPM(model=model, timesteps=1000).to(device)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        y = torch.randint(0, num_classes, (args.batch_size,), device=device)
        batch = torch.stack([images[label] for label in y.tolist()])
        loss = ddpm(batch, y)
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
            "num_classes": num_classes,
            "class_names": class_names,
        },
        args.output,
    )
    print(f"saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
