# MatAugment
**Data Augmentation Benchmark for Micrographs and DFT Structures**

(This is a work in progress and currently in the middle of 'code changes' and 'initial testing')


This repository provides a clean, reproducible framework for benchmarking **data augmentation methods** on:

- Ultrahigh Carbon Steel (UHCS) micrograph images
- NIST DFT benchmark CIF structures

Problem Stateent: Since microscopy and DFT datasets are often small, noisy, or expensive to collect, the goal of this project is to develop a unified data-augmentation framework (“MatAugment”) to systematically evaluate how geometric, noise-based, and generative augmentations affect model generalization for both DFT and Micrograph datasets, with the goal of improving robustness and performance in materials-informatics workflows.


## Overview

### **Micrographs (UHCS)**
A convolutional autoencoder is trained to reconstruct microstructures under different augmentations:
- `none` (baseline)
- `basic` (rotation + flips)
- `noise`
- `crop`
- `combined`

Performance is reported using:
- **MSE**
- **SSIM**

---

### **DFT Atomic Structures**
A point-cloud autoencoder reconstructs normalized atomic coordinates from CIF files.  
Augmentations include:
- `none`
- `jitter`
- `rotation`
- `dropout`
- `combined`

Performance is measured using:
- **Coordinate reconstruction MSE**

---

## Project Structure
mataugment
─ micrograph/ # Micrograph AE, augmentations, training & evaluation
─ dft/ # DFT point-cloud AE, augmentations, training & evaluation
─ utils/ # Utilities for visualization


---

## Running the Micrograph Experiments
**Train**
```bash
from mataugment.micrograph.train import train_autoencoder
print("None : ")

train_autoencoder(
    data_dir="/Users/gowri_dhara/CAMMP_Project/data/micrographs",
    out_dir="/Users/gowri_dhara/CAMMP_Project/outputs/micro_none",
    aug_name="none",
    epochs=20,
    batch_size=16
)
```

**Evaluate**
```bash
from mataugment.micrograph.evaluate import evaluate_reconstruction
print("Crop : ")
print(
    evaluate_reconstruction(
        "/Users/gowri_dhara/CAMMP_Project/outputs/micro_crop/ae_crop_ep020.pt",
        "/Users/gowri_dhara/CAMMP_Project/data/micrographs",
        "crop"
    )
)
```
---

## Purpose

This benchmark enables:

- Understanding how augmentations affect feature stability

- Comparing reconstruction robustness across augmentation types

- Providing a baseline for future supervised or generative models

The framework is modular and easy to extend with new augmentations or architectures.


