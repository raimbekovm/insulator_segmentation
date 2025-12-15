# Insulator Segmentation

Deep learning model for automatic insulator segmentation on power line images captured by UAVs (drones).

## Overview

This project implements a U-Net architecture with ResNet34 encoder for semantic segmentation of insulators in aerial power line inspection images. The model achieves **0.9895 Dice coefficient** on validation data.

## Problem Statement

Power line inspection requires identifying insulators on thousands of transmission tower photos. Manual inspection is time-consuming and labor-intensive. This automated solution enables rapid defect detection and significantly reduces inspection time.

## Architecture

- **Model**: U-Net with pretrained ResNet34 encoder
- **Framework**: PyTorch
- **Input**: 512×512 RGB images
- **Output**: Binary segmentation masks
- **Techniques**: Transfer Learning, Data Augmentation, Test-Time Augmentation (TTA)

## Dataset

- **Training**: 7,000 images with masks
- **Test**: 4,000 images
- **Format**: JPG (images), PNG (masks)
- **Split**: 85% train (5,950), 15% validation (1,050)

## Performance

| Metric | Value |
|--------|-------|
| Validation Dice | 0.9895 |
| Training Epochs | 15 (with early stopping) |
| Batch Size | 16 |
| Image Size | 512×512 |
| Training Time | ~13 hours on 2×Tesla T4 |

## Key Features

- **Transfer Learning**: Uses ImageNet pretrained ResNet34 encoder
- **Mixed Precision Training**: FP16 for faster training
- **Data Augmentation**: Flips, rotations, brightness/contrast adjustments
- **Test-Time Augmentation**: 4× augmentation for improved predictions
- **Threshold Optimization**: Automatic threshold tuning on validation set
- **Early Stopping**: Prevents overfitting (patience=7 epochs)

## Project Structure

```
01_insulator_segmentation/
├── README.md                       # This file
├── insulator_segmentation.ipynb    # Main training notebook
├── requirements.txt                # Python dependencies
└── samples/                        # Sample images
    ├── images/                     # Sample input images
    └── masks/                      # Sample ground truth masks
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Open `insulator_segmentation.ipynb` and run all cells. The notebook includes:

1. Data loading and preprocessing
2. Model architecture definition
3. Training loop with validation
4. Threshold optimization
5. Test predictions with TTA

### Key Hyperparameters

```python
IMG_SIZE = 512
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
ENCODER = 'resnet34'
USE_TTA = True
```

## Model Details

### U-Net Architecture

The U-Net consists of:
- **Encoder**: ResNet34 (5 downsampling levels)
- **Decoder**: 5 upsampling levels with skip connections
- **Output**: Single channel (binary mask)

### Loss Function

Combined loss for better convergence:
```
Loss = 0.5 × BCE + 0.5 × (1 - Dice)
```

### Data Augmentation

Training augmentations:
- Horizontal/vertical flips
- Random 90° rotations
- Shift, scale, rotate (±45°)
- Brightness/contrast adjustments
- Gaussian noise/blur

## Results

The model achieves high accuracy in segmenting insulators across various conditions including different lighting, angles, and weather conditions.

See `samples/` directory for example inputs and masks.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- See `requirements.txt` for complete dependencies

## Future Improvements

- Experiment with larger encoders (ResNet50, EfficientNet-B4)
- Implement ensemble of multiple models
- Add multi-scale training
- Explore attention mechanisms

## Acknowledgments

- Dataset from Clean Code Cup 2022 - Problem A
- U-Net architecture: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- ResNet: [He et al., 2015](https://arxiv.org/abs/1512.03385)

---

[← Back to Course](../)
