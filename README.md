# Stance-Aware Hierarchical Structural Weighting for Rumor Detection

This repository contains the code and experimental scripts for the paper:

**Stance-Aware Hierarchical Structural Weighting for Social Media Rumor Detection**  
(Chinese manuscript version: *基于立场感知与层级结构加权的社交媒体谣言检测方法*)

---

## 🔥 Overview

Rumor detection on social media is challenging due to noisy deep replies in propagation trees.
This project proposes a **stance-aware** and **depth-sensitive** rumor detection framework:

- **Stance-Aware Node Representation**: encode each node with a pretrained language model (BERT).
- **Hierarchical Structural Weighting (Evidence Decay)**: assign smaller weights to deeper replies to suppress noise.
- **Tree-level Aggregation**: aggregate node representations into a single tree representation for classification.

We provide reproducible scripts for experiments on:
- **Twitter15 / Twitter16** (cross-dataset evaluation)
- **DRWeibo** (in-domain evaluation, mean±std over multiple seeds)
- 
---
data link: https://github.com/Derrickqi/Rumor-detection/releases
---

## 📌 Key Features

- ✅ Root-only baseline (source tweet only)
- ✅ Uniform aggregation baseline
- ✅ Evidence Decay aggregation (depth-weighted pooling)
- ✅ Multi-seed evaluation with mean ± std
- ✅ Preprocessed `.pt` datasets for fast training

---



