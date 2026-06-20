# DSOD: Debiased Source-Free Object Detection

**Injecting debiased foundation-model features for cross-domain detection.**

> This work builds upon [DRU](https://github.com/lbktrinh/DRU) (ECCV 2024) and introduces **DSOD (Debiased Source-Free Object Detection)**, which injects **debiased foundation-model features** (DINOv2) into the Mean-Teacher framework to boost cross-domain detection performance in the source-free setting.
>
> **Accepted to _Pattern Recognition_, Vol. 26 (2026).**

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Weights](#pretrained-weights)
- [Usage](#usage)
  - [1. Source-only pretraining](#1-source-only-pretraining)
  - [2. DSOD adaptation training (with DINOv2 fusion)](#2-dsod-adaptation-training-with-dinov2-fusion)
  - [3. Evaluation](#3-evaluation)
- [Key Options](#key-options)
- [Fusion Strategies](#fusion-strategies)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

**Source-Free Object Detection (SFOD)** adapts a detector to a target domain *without access to source data*. DRU tackles this with a Dynamic Retraining-Updating Mean Teacher. **DSOD** extends DRU by injecting features from the **DINOv2** foundation model — which is trained on broad, diverse data and is largely domain-invariant — into the CNN backbone's multi-scale features. This *debiases* the source-domain-biased CNN features and improves generalization to the target domain.

```
Input image ─┬─► ResNet50 backbone ─► CNN multi-scale features (layer2/3/4)
              │                                  │
              └─► DINOv2 backbone ─► patch tokens ─┤
                                                 ▼
                                       Feature Fusion (fuse)
                                                 │
                                                 ▼
                                     Deformable Transformer ─► Detection output
```

The teacher model generates pseudo-labels on the target domain; the student is trained on both the pseudo-labels and a masked-view consistency loss, with EMA updating the teacher.

---

## Installation

### Requirements

- Linux, CUDA >= 11.1, GCC >= 8.4
- Python >= 3.8
- PyTorch >= 1.10.1, torchvision >= 0.11.2

### Install dependencies

```bash
pip install -r requirements.txt
```

### Compile Deformable DETR CUDA operators

```bash
cd ./models/ops
sh ./make.sh
# unit test (all checks should be True)
python test.py
```

### DINOv2 backbone

The DINOv2 implementation is included under `dinov2/` (local, loaded via `torch.hub.load(..., source='local')`). Place the pretrained DINOv2 checkpoint at:

```
weights/dinov2_vitb14_reg4_pretrain.pth
```

(used model: `dinov2_vitb14_reg`, feature dim 768). See [Pretrained Weights](#pretrained-weights).

---

## Dataset Preparation

DSOD uses 3 standard SFOD benchmarks:

| Benchmark   | Source domain     | Target domain                       |
| ----------- | ----------------- | ----------------------------------- |
| `city2foggy`| Cityscapes        | Foggy Cityscapes (fog level 0.02)   |
| `sim2city`  | Sim10k            | Cityscapes (`car` only)             |
| `city2bdd`  | Cityscapes        | BDD100k-daytime                     |

Download raw data from the official sites: [Cityscapes](https://www.cityscapes-dataset.com/), [Foggy Cityscapes](https://www.cityscapes-dataset.com/), [Sim10k](https://fcav.engin.umich.edu/projects/driving-in-the-matrix), [BDD100k](https://bdd-data.berkeley.edu/).

COCO-style annotations (provided by [MRT-release](https://github.com/JeremyZhao1998/MRT-release)) can be downloaded [here](https://drive.google.com/file/d/1LB0wK9kO3eW8jpR2ZtponmYWe9x2KSiU/view?usp=sharing).

Organize data as:

```
[data_root]
└── cityscapes
    ├── annotations
    │   ├── cityscapes_train_cocostyle.json
    │   ├── cityscapes_train_caronly_cocostyle.json
    │   ├── cityscapes_val_cocostyle.json
    │   └── cityscapes_val_caronly_cocostyle.json
    └── leftImg8bit/{train,val}
└── foggy_cityscapes
    ├── annotations
    │   ├── foggy_cityscapes_train_cocostyle.json
    │   └── foggy_cityscapes_val_cocostyle.json
    └── leftImg8bit_foggy/{train,val}
└── sim10k
    ├── annotations
    │   └── sim10k_train_cocostyle.json
    └── JPEGImages
└── bdd10k
    ├── annotations
    │   ├── bdd100k_daytime_train_cocostyle.json
    │   └── bdd100k_daytime_val_cocostyle.json
    └── images
```

---

## Pretrained Weights

Two kinds of pretrained weights are needed:

1. **DINOv2 backbone** — `dinov2_vitb14_reg4_pretrain.pth`, placed under `weights/`. Available from the [DINOv2 repository](https://github.com/facebookresearch/dinov2).
2. **Source-only detector** — produced by step 1 below, e.g. `city2foggy_source_only_29_53.pth`. DRU's original source-only weights can be found in the [DRU release](https://github.com/lbktrinh/DRU).

---

## Usage

The training pipeline has 3 stages: **source-only pretraining** → **DSOD adaptation** → **evaluation**.

All scripts are under `configs/def-detr-base/<benchmark>/`. Edit a script to set your own `DATA_ROOT`, `OUTPUT_DIR`, and the `--resume` checkpoint path before running.

### 1. Source-only pretraining

Train a detector on the labeled source domain (no target adaptation yet):

```bash
sh configs/def-detr-base/city2foggy/source_only.sh
```

This produces a source-only checkpoint (e.g. `city2foggy_source_only_29_53.pth`), which is the starting point for adaptation.

### 2. DSOD adaptation training (with DINOv2 fusion)

This is the **core of DSOD**. Inject DINOv2 features and run Mean-Teacher adaptation. Two modes are supported:

**a) Standard Mean-Teacher + DINOv2 fusion** (`teaching_standard_dino`):

```bash
sh configs/def-detr-base/city2foggy/teaching_standard_dino.sh
```

**b) DRU masked teaching + DINOv2 fusion** (recommended, `teaching_mask_dino`):

```bash
sh configs/def-detr-base/city2foggy/teaching_mask_dino.sh
```

Example command (city2foggy, DRU mask mode + DINOv2):

```bash
torchrun --nproc_per_node=8 main.py \
    --enable_dino \
    --enable_smooth \
    --dino_weight 0.4 \
    --backbone resnet50 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_classes 9 \
    --dropout 0.0 \
    --data_root ./dataset \
    --source_dataset cityscapes \
    --target_dataset foggy_cityscapes \
    --batch_size 8 \
    --eval_batch_size 8 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_linear_proj 2e-5 \
    --alpha_ema 0.999 \
    --epoch 30 \
    --epoch_lr_drop 80 \
    --enable_feature_alignment \
    --mode teaching_mask \
    --threshold 0.3 \
    --dynamic_update \
    --max_update_iter 5 \
    --only_class_loss \
    --use_pseudo_label_weights \
    --output_dir ./outputs/def-detr-base/city2foggy/teaching_mask_dino \
    --resume ./city2foggy_source_only_29_53.pth
```

The same pattern applies to the other benchmarks — see `configs/def-detr-base/sim2city/` and `configs/def-detr-base/city2bdd/`.

### 3. Evaluation

```bash
sh configs/def-detr-base/city2foggy/evaluation_teaching_mask.sh
```

or set `--mode eval` with the trained checkpoint via `--resume`.

---

## Key Options

| Flag | Type | Description |
| --- | --- | --- |
| `--enable_dino` | flag | Enable DINOv2 feature injection (core of DSOD). |
| `--dino_weight` | float | Fusion weight for DINOv2 features (e.g. `0.4`). |
| `--dino_alpha` | float | EMA coefficient for the DINO factor (default `1.0`). |
| `--fuse_type` | str | Fusion strategy, see [Fusion Strategies](#fusion-strategies). |
| `--enable_smooth` | flag | Smoothly ramp up the DINO factor during training. |
| `--enable_feature_alignment` | flag | Align fused features between teacher/student. |
| `--test_stability` | flag | Run stability test + binary search for the DINO factor. |
| `--mode` | str | `single_domain` / `teaching_standard` / `teaching_mask` / `eval`. |
| `--dynamic_update` | flag | Enable DRU's dynamic retraining-updating. |
| `--max_update_iter` | int | Max iterations for dynamic update (default `5`). |
| `--use_pseudo_label_weights` | flag | Weight pseudo-labels by confidence. |
| `--threshold` | float | Confidence threshold for pseudo-labels (default `0.3`). |
| `--alpha_ema` | float | Teacher EMA coefficient (default `0.999`). |

---

## Fusion Strategies

DSOD supports multiple ways to fuse DINOv2 features with CNN features, selectable via `--fuse_type`:

| `fuse_type` | Description |
| --- | --- |
| `add` | Direct addition of projected DINOv2 features. |
| `cat` | Concatenate then project to CNN channel dim. |
| `cat_add` | Concatenate, project, and residual-add to CNN features. |
| `gate_add` | Channel gate from DINOv2 features, then gated addition. |
| `gate_add_cnn` | Gate computed from projected DINOv2 features. |
| `grl_gate_add` | Gate with a Gradient Reversal Layer for domain debiasing. |
| `residual_add` | Fuse the *residual* (DINOv2 − CNN) with a gate. |
| `ch_add` | Channel-wise gated addition. |
| `sp_ch_add` | Spatial × channel gated addition. |
| `adaptive_add` | Adaptive scale-normalized addition. |
| `mul` | Multiplicative blending. |
| `attn` | Cross-attention fusion (CNN queries attend to DINOv2 keys/values). |

---

## Acknowledgements

This project is built upon:

- [DRU](https://github.com/lbktrinh/DRU) (ECCV 2024) — the base Mean-Teacher + dynamic retraining framework.
- [MRT-release](https://github.com/JeremyZhao1998/MRT-release) — dataset annotations and SFOD baselines.
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) — the detector architecture.
- [DINOv2](https://github.com/facebookresearch/dinov2) — the foundation vision model.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{CAI2026113545,
title = {Towards unbiased source-free object detection via vision foundation models},
journal = {Pattern Recognition},
volume = {179},
pages = {113545},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.113545},
url = {https://www.sciencedirect.com/science/article/pii/S003132032600511X},
author = {Zhi Cai and Yingjie Gao and Yanan Zhang and Xinzhu Ma and Di Huang},
keywords = {Source-free object detection, VFM, Knowledge distillation},
abstract = {Source-Free Object Detection (SFOD) has garnered much attention in recent years by eliminating the need of source-domain data in cross-domain tasks, but existing SFOD methods suffer from the Source Bias problem, i.e. the adapted model remains skewed towards the source domain, leading to poor generalization and error accumulation during self-training. To overcome this challenge, we propose Debiased Source-free Object Detection (DSOD), a novel VFM-assisted SFOD framework that can effectively mitigate source bias with the help of powerful VFMs. Specifically, we propose Unified Feature Injection (UFI) module that integrates VFM features into the CNN backbone through Simple-Scale Extension (SSE) and Domain-aware Adaptive Weighting (DAAW). Then, we propose Semantic-aware Feature Regularization (SAFR) that constrains feature learning to prevent overfitting to source domain characteristics. Furthermore, we propose a VFM-free variant, termed DSOD-distill for computation-restricted scenarios through a novel Dual-Teacher distillation scheme. Extensive experiments on multiple benchmarks demonstrate that DSOD outperforms state-of-the-art SFOD methods, achieving 48.1% AP on Normal-to-Foggy weather adaptation, 39.3% AP on Cross-scene adaptation, and 61.4% AP on Synthetic-to-Real adaptation.}
}
```

Please also cite the base DRU work:

```bibtex
@inproceedings{trinh2024dru,
  title     = {Dynamic Retraining-Updating Mean Teacher for Source-Free Object Detection},
  author    = {Trinh, Le Ba Khanh and Nguyen, Huy-Hung and Pham, Long Hoang and Tran, Duong Nguyen-Ngoc and Jeon, Jae Wook},
  booktitle = {ECCV},
  pages     = {328--344},
  year      = {2024}
}
```
