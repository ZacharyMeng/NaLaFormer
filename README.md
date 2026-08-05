# Norm×Direction: Restoring the Missing Query Norm in Vision Linear Attention [ICML 2026]

🚀 Welcome to the repo of **NaLaFormer**!

This repo contains the official **PyTorch** code for NaLaFormer.

[![arXiv](https://img.shields.io/badge/Arxiv-2506.21137-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.21137)

## Introduction

### Motivation

Linear attention mitigates the quadratic complexity of softmax attention but suffers from a critical loss of expressiveness. We identify two primary causes: (1) The normalization operation cancels the query norm, which breaks the correlation between a query's norm and the spikiness (entropy) of the attention distribution as in softmax attention. (2) Standard techniques for enforcing non-negativity cause destructive information loss by nullifying valid inner-product interactions.

To address these challenges, we propose **NaLaFormer**, which achieves a superior balance between expressive capability and efficiency.

### Method

We introduce NaLaFormer, a novel linear attention mechanism built upon a **norm×direction (ND)** decomposition of the query and key vectors. We leverage each component to solve a distinct problem:

- **Query-norm-aware feature map**: The query norm is injected into our kernel to create a query-norm-aware map that restores the attention distribution's spikiness.
- **Cosine direction similarity**: The direction vectors are processed by a geometric, cosine-based similarity metric that guarantees non-negativity while preserving the rich, fine-grained information of the inner product.

### Results

- Comparison of different models on ImageNet-1K.

| Model | Params | FLOPs | Top-1 Acc (%) |
| :---: | :---: | :---: | :---: |
| NaLaFormer-XT | 8M | 1.0G | 79.1 |
| NaLaFormer-T | 15M | 2.7G | 82.6 |
| NaLaFormer-S | 26M | 5.1G | 84.3 |
| NaLaFormer-B | 52M | 12G | 85.2 |
| NaLaFormer-L | 95M | 18G | 85.7 |

## Dependencies

- Python 3.9+
- PyTorch == 2.4.0
- torchvision == 0.19.0
- numpy
- timm >= 0.4.12
- fvcore
- einops

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Pretrained Models

Pre-trained weights will be released soon. The following models are supported in this repo:

| model | Reso | acc@1 | model name |
| :---: | :---: | :---: | :---: |
| NaLaFormer-XT | 224² | 79.1 | `NALAFORMER_XT` |
| NaLaFormer-T | 224² | 82.6 | `NALAFORMER_T` |
| NaLaFormer-S | 224² | 84.3 | `NALAFORMER_S` |
| NaLaFormer-B | 224² | 85.2 | `NALAFORMER_B` |
| NaLaFormer-L | 224² | 85.7 | `NALAFORMER_L` |

Evaluate one model on ImageNet:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model NALAFORMER_XT \
    --data-path <imagenet-path> \
    --output_dir <output-path> \
    --eval --dist-eval
```

## Train Models from Scratch

**To train NaLaFormer on ImageNet from scratch, see `pretrain.sh` and run:**

```shell
export DATA=/path/to/imagenet
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
bash pretrain.sh
```

You can also specify the model variant via `--model`, e.g. `NALAFORMER_T`, `NALAFORMER_S`, `NALAFORMER_B`, `NALAFORMER_L`.

## Citation

If you find this repo helpful, please consider citing us.

```latex
@inproceedings{meng2026nalaformer,
  title={Norm$\times$Direction: Restoring the Missing Query Norm in Vision Linear Attention},
  author={Weikang Meng and Yadan Luo and Liangyu Huo and Yingjian Li and Yaowei Wang and Xin Li and Zheng Zhang},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```
