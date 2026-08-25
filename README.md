<div align="center">

<img src="assets/logo.png" width="110">

# NaLaFormer

**Norm × Direction: Restoring the Missing Query Norm in Vision Linear Attention**

[![arXiv](https://img.shields.io/badge/arXiv-2506.21137-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2506.21137)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b8bbe.svg)](https://arxiv.org/abs/2506.21137)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/ZacharyMeng/NaLaFormer?style=social)](https://github.com/ZacharyMeng/NaLaFormer/stargazers)

[Weikang Meng](https://github.com/ZacharyMeng), Yadan Luo, Liangyu Huo, Yingjian Li, Yaowei Wang, Xin Li, Zheng Zhang

*International Conference on Machine Learning (ICML), 2026*

</div>

---

## 📰 News

- **[2026.07]** 🎉 NaLaFormer is accepted by **ICML 2026**!
- **[2026.06]** 🚀 Code and ImageNet-1K training recipes are released.
- **[2026.06]** 📄 Paper available on [arXiv](https://arxiv.org/abs/2506.21137).

## 📖 Introduction

Linear attention mitigates the quadratic complexity of softmax attention but suffers from a critical loss of expressiveness. Applying a **norm–direction decomposition** $\mathbf{q}_t = \|\mathbf{q}_t\|\, d(\mathbf{q}_t)$, $\mathbf{k}_i = \|\mathbf{k}_i\|\, d(\mathbf{k}_i)$ to linear attention,

$$\mathrm{LinearAttn}_t = \frac{\phi(\mathbf{q}_t)\sum_{i=1}^{N}\phi(\mathbf{k}_i)^\top\mathbf{v}_i}{\phi(\mathbf{q}_t)\sum_{j=1}^{N}\phi(\mathbf{k}_j)^\top} = \frac{\|\phi(\mathbf{q}_t)\|\; d(\phi(\mathbf{q}_t))\sum_{i=1}^{N}\|\phi(\mathbf{k}_i)\|\, d(\phi(\mathbf{k}_i))^\top\mathbf{v}_i}{\|\phi(\mathbf{q}_t)\|\; d(\phi(\mathbf{q}_t))\sum_{j=1}^{N}\|\phi(\mathbf{k}_j)\|\, d(\phi(\mathbf{k}_j))^\top},$$

exposes a critical **norm cancellation**: the query norm $\|\phi(\mathbf{q}_t)\|$ appears in both the numerator and the denominator, so it is *canceled out* by the division. As a result, only *key* norms influence linear attention outputs — the model becomes **norm-unaware** w.r.t. queries, and the correlation between a query's norm and the spikiness (entropy) of its attention distribution, which softmax attention naturally enjoys, is destroyed. Meanwhile, standard element-wise tricks for enforcing non-negativity (e.g. $\mathrm{ReLU}$, $1+\mathrm{ELU}$) further nullify valid signed inner-product interactions.

<div align="center">
<img src="assets/fig1_motivation.png" width="85%">
<p><em>Correlation between entropy and vector norm: query norms correlate strongly with entropy
in softmax attention (top), while key norms exhibit only weak correlation (bottom).</em></p>
</div>

## 🛠️ Method

We propose **NaLaFormer**, a linear attention built upon a **Norm × Direction (ND)** decomposition of queries and keys, where each component fixes one problem:

- **Query-norm-aware feature map** — the query norm $\|\mathbf{q}\|$ is re-injected into the kernel through a $\tanh$-bounded power function, restoring attention spikiness while keeping the map norm-aware and numerically stable.
- **Cosine inhibit** — direction vectors are compared via a cosine-based similarity $[\cos(\cdot), \sin(\cdot)]$, which guarantees non-negativity and preserves the norm, without discarding fine-grained inner-product information.

<div align="center">
<img src="assets/fig2_framework.png" width="95%">
<p><em>The overall framework of NaLaFormer: (a) gated linear attention with the ND-decomposed kernel,
(b) query-norm-aware spikiness, and (c) cosine inhibit for non-negativity.</em></p>
</div>

## 📊 Results

ImageNet-1K classification at 224² resolution, trained from scratch:

| Model | Params | FLOPs | Top-1 Acc (%) |
| :---: | :---: | :---: | :---: |
| NaLaFormer-XT | 8M | 1.0G | 79.1 |
| NaLaFormer-T | 15M | 2.7G | 82.6 |
| NaLaFormer-S | 26M | 5.1G | 84.3 |
| NaLaFormer-B | 52M | 12G | 85.2 |
| NaLaFormer-L | 95M | 18G | 85.7 |

<div align="center">
<img src="assets/fig6_efficiency.png" width="52%">
<p><em>Efficiency analysis: Accuracy vs. FLOPs curves on ImageNet-1K.</em></p>
</div>

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/ZacharyMeng/NaLaFormer.git
cd NaLaFormer

conda create -n nalaformer python=3.9 -y
conda activate nalaformer

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Data Preparation

The ImageNet-1K dataset should be organized as follows:

```
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   └── ...
    └── ...
```

### Evaluation

Evaluate a pre-trained model on the ImageNet validation set (pre-trained weights will be released soon):

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model NALAFORMER_XT \
    --data-path <imagenet-path> \
    --output_dir <output-path> \
    --eval --dist-eval
```

### Training from Scratch

See `pretrain.sh` for the full recipe, or simply run:

```bash
export DATA=/path/to/imagenet
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC=8
bash pretrain.sh
```

Specify other variants via `--model`, e.g. `NALAFORMER_T`, `NALAFORMER_S`, `NALAFORMER_B`, `NALAFORMER_L`.

## 📁 Repository Structure

```
NaLaFormer/
├── NaLaformer/
│   ├── NALA.py        # NaLaFormer backbone & Norm-Aware Linear Attention
│   ├── main.py        # training / evaluation entry point
│   ├── engine.py      # train & eval loops
│   ├── datasets.py    # ImageNet data pipeline & augmentations
│   ├── losses.py      # soft-target / label-smoothing criterion
│   ├── samplers.py    # RASampler for distributed training
│   ├── utils.py       # EMA, checkpointing, metrics, misc.
│   └── pretrain.sh    # ImageNet-1K pre-training recipe
├── assets/            # logo & paper figures used in this README
├── requirements.txt
└── README.md
```

## ✅ TODO

- [x] Release ImageNet-1K training & evaluation code
- [ ] Release pre-trained weights
- [ ] Downstream tasks: object detection & semantic segmentation
- [ ] Throughput / latency benchmark scripts

## 📝 Citation

If you find this repo helpful, please consider citing us:

```bibtex
@inproceedings{meng2026nalaformer,
  title     = {Norm$\times$Direction: Restoring the Missing Query Norm in Vision Linear Attention},
  author    = {Weikang Meng and Yadan Luo and Liangyu Huo and Yingjian Li and Yaowei Wang and Xin Li and Zheng Zhang},
  booktitle = {International Conference on Machine Learning},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This codebase builds upon the excellent open-source projects [timm](https://github.com/huggingface/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer). We thank the authors for their great work.

## 📄 License

This project is released under the [MIT License](LICENSE).
