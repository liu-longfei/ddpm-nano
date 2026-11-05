# 极简单图 DDPM

这个项目用一个很小的 U-Net 学习一张图片，训练后用 50 步 DDIM 快速采样。默认训练和保存的图片分辨率是 `128×128`；U-Net 是全卷积的，采样时也可以覆盖为其他 4 的倍数分辨率。

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
python3 train.py --image data/target.png --image-size 128 --steps 3000
```

CPU 上通常需要更长时间；有 GPU 会更快。训练会生成 `ddpm.pt`，并把训练分辨率写入 checkpoint。

## 推理

```bash
python3 sample.py --checkpoint ddpm.pt --output generated.png --steps 50

# 使用已有 32x32 checkpoint 采样为 128x128（仅改变输出尺寸，不改变模型权重）
python3 sample.py --checkpoint ddpm.pt --image-size 128 --output generated.png --steps 50
```

如果想更接近原图，可以把训练步数提高到 5000～8000；采样步数也可尝试 100。

## 极简 Conditional DDPM

这是类别条件版本：模型在 timestep embedding 上加入 class label embedding。默认会把 `data/target.png` 当作类别 0 训练；如果有多类数据，把图片放进 `data/conditional/`，按文件名排序后分别作为类别 0、1、2……

### 训练

```bash
python3 train_condition.py --image-size 128 --steps 3000
```

多类示例：

```bash
python3 train_condition.py --data-dir data/conditional --image-size 128 --steps 3000
```

训练会生成 `conditional_ddpm.pt`。

### 推理

```bash
python3 sample_condition.py --checkpoint conditional_ddpm.pt --class 0 --steps 50

# 覆盖采样分辨率时，也必须是 4 的倍数
python3 sample_condition.py --checkpoint conditional_ddpm.pt --class 0 --image-size 128 --steps 50
```
