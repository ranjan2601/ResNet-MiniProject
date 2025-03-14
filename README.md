# ResNet on CIFAR-10: Lightweight Model Training and Optimization

This repository contains a deep learning project to train a lightweight ResNet model on the CIFAR-10 dataset, achieving strong performance with fewer than 5 million parameters. The project includes an initial training phase followed by optimization techniques such as a cosine annealing scheduler, Mixup augmentation, and Test-Time Augmentation (TTA) for improved predictions.

## Project Overview

- **Dataset**: CIFAR-10 (10 classes, 50,000 training images, 10,000 test images, 32x32 RGB)
- **Model**: A custom `ResNetSmall` architecture based on ResNet with `BasicBlock`, designed to have fewer than 5 million parameters (approximately 2.74M parameters)
- **Goal**: Train an efficient model with high accuracy on CIFAR-10, then optimize it for better generalization and performance

## Model Details

- **Architecture**: `ResNetSmall` with `BasicBlock` (4 layers: [4, 4, 4, 3] blocks)
- **Parameter Count**: ~4.74M trainable parameters
- **Input**: 32x32 RGB images (3 channels)
- **Output**: 10-class softmax probabilities

## Requirements

- Python 3.8+
- PyTorch 1.9+ (`torch`, `torchvision`)
- Additional libraries: `torchinfo`, `numpy`, `matplotlib`

## Training and Optimization Phases

### Initial Training
- **Optimizer:** SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **Scheduler:** MultiStepLR (milestones=[50, 75], gamma=0.1)
- **Loss:** Cross-Entropy Loss
- **Epochs:** 100
- **Data Augmentation:** Random crop, horizontal flip, normalization
- **Model Saving:** Best model based on test accuracy was saved


#### Mixup
- Added Mixup augmentation (`alpha=0.2`) to the training loop
- Applied during training to mix inputs and targets

#### Test-Time Augmentation (TTA)
- Generated predictions by averaging over multiple augmented versions of test images

## Results
- **Initial Model:** Achieved ~94% test accuracy.
- **Optimized Model:** Improved to ~95.5% test accuracy with Mixup, and TTA (update with actual results)

