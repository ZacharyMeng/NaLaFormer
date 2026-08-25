# NaLaFormer Model Zoo

All models are trained on ImageNet-1K at 224² resolution for 300 epochs.

## ImageNet-1K Results

| Model | Params | FLOPs | Top-1 Acc (%) | Config name | Weights |
| :---: | :---: | :---: | :---: | :---: | :---: |
| NaLaFormer-XT | 8M | 1.0G | 79.1 | `NALAFORMER_XT` | coming soon |
| NaLaFormer-T | 15M | 2.7G | 82.6 | `NALAFORMER_T` | coming soon |
| NaLaFormer-S | 26M | 5.1G | 84.3 | `NALAFORMER_S` | coming soon |
| NaLaFormer-B | 52M | 12G | 85.2 | `NALAFORMER_B` | coming soon |
| NaLaFormer-L | 95M | 18G | 85.7 | `NALAFORMER_L` | coming soon |

## Architecture Configurations

| Config | XT | T | S | B | L |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Embed dims | [32, 64, 128, 384] | [64, 128, 256, 512] | [64, 128, 320, 512] | [96, 192, 384, 512] | [96, 192, 448, 640] |
| Depths | [2, 2, 2, 2] | [2, 2, 6, 2] | [3, 5, 9, 3] | [4, 6, 12, 6] | [4, 7, 19, 8] |
| Num heads | [1, 2, 4, 12] | [1, 2, 4, 8] | [1, 2, 5, 8] | [1, 2, 6, 8] | [1, 2, 7, 10] |
| MLP ratios | 3.5 | 4.0 | 3.5 | 4.0 | 3.5 |
| Drop path rate | 0.10 | 0.10 | 0.15 | 0.40 | 0.55 |
| Projection dim | 1024 | 1024 | 1024 | 1024 | 1024 |

## Usage

```python
import torch
from NaLaformer.NALA import NALAFORMER_S

model = NALAFORMER_S()
x = torch.randn(1, 3, 224, 224)
out = model(x)   # (1, 1000)
```

Or via `timm`:

```python
import timm
from NaLaformer import NALA  # registers all variants

model = timm.create_model("NALAFORMER_S", pretrained=False, num_classes=1000)
```
