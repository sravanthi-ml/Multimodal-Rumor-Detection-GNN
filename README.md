# Multimodal-Rumor-Detection-GNN
Official implementation and dataset resources for the paper: "Multimodal Deep Learning and Graph-Aware Architectures for Robust Rumor Detection on Social Media" (Scientific Reports, Minor Revision).
# Multimodal Deep Learning and Graph-Aware Architectures for Robust Rumor Detection on Social Media

This repository contains the official implementation and experimental resources for the paper submitted to *Scientific Reports*.

## 📌 Overview

This work proposes a multimodal deep learning framework integrating:

- Transformer-based textual encoding
- Vision Transformer (ViT) image encoding
- Metadata feature modeling
- Graph Attention Networks (GAT) for rumor propagation modeling
- Contrastive cross-modal alignment

The model is evaluated on benchmark rumor detection datasets including PHEME, Twitter15, and Twitter16.

---

## 📂 Repository Structure

- `/data/` – Dataset folders
- `/src/` – Training and evaluation scripts
- `/models/` – Saved trained models
- `requirements.txt` – Python dependencies

---

## 📊 Datasets

The following publicly available datasets were used:

- **PHEME Dataset**
  https://figshare.com/articles/dataset/PHEME_dataset/4010619

- **Twitter15 & Twitter16**
  https://www.kaggle.com/datasets/syntheticprogrammer/rumor-detection-acl-2017

All datasets are publicly accessible and used strictly for academic research purposes.

---

## ⚙️ Requirements

Python 3.9+

Install dependencies using:

