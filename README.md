# Proximal Dehazeformer: Vision Transformer with Learnable Priors for Image Dehazing

## Overview

This repository contains the implementation of the Proximal Dehazeformer, a novel approach for image dehazing using a Vision Transformer (ViT) architecture with learnable priors. This method addresses the limitations of traditional image dehazing techniques and convolutional neural networks (CNNs) by leveraging the ability of ViTs to capture long-range dependencies within an image, combined with learnable haze priors for enhanced performance.

## Features

- **Vision Transformer with Learnable Priors**: Integrates haze priors within the ViT architecture to model complex haze conditions.
- **Proximal Dehazeformer Block**: Modified architecture for more efficient latent representations, resulting in faster processing times.
- **State-of-the-Art Performance**: Achieves superior results on datasets with non-homogeneous haze.
- **Dataset Diversity**: Trained and evaluated on a combined dataset from NH-Haze, BeDDe, and RESIDE-Outdoor.

## Dataset

The model was trained on a dataset containing 400 hazy and corresponding clear images, compiled from three established sources:

- **NH-Haze**: 55 real-world outdoor scenes with non-uniform haze distribution.
- **BeDDe**: 23 different scenes with a total of 208 paired fogged/clear images.
- **RESIDE-Outdoor**: Widely used for evaluating single image dehazing algorithms, containing 137 images for training and testing.

## Preprocessing

- **CLAHE (Contrast-Limited Adaptive Histogram Equalization)**: Enhances image quality by preserving color fidelity and details without introducing artifacts.
- **LAB Color Space**: CLAHE applied to the luminance channel in LAB color space for better visual outcomes.

## Model Architecture

### Baseline DehazeFormer

- Utilizes the Swin Transformer architecture.
- Incorporates convolutional layers, attention mechanisms, and spatial information aggregation schemes.

### Proximal Dehazeformer

- Introduces a Proximal Net block with learnable priors.
- Enhances encoder-decoder architecture with additional convolution layers and ReLU activation.

## Experimentation

- **Training**: Conducted on Kaggle’s GPUs, using AdamW optimizer with a learning rate of 2e-4 for 60 epochs.
- **Evaluation Metrics**: Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM) for quantitative assessment.

## Results

- **Quantitative**: Demonstrated competitive PSNR and SSIM values against baseline models on BeDDe and RESIDE datasets.
- **Qualitative**: Evaluations indicate improved haze removal and visually appealing dehazed images.

## Installation

To run this project locally:

```bash
git clone https://github.com/your-username/Vision-Transformer-Image-Dehazing.git
cd Vision-Transformer-Image-Dehazing
pip install -r requirements.txt
