# ResNet on CIFAR-10: Lightweight Model Training and Optimization

This repository contains a deep learning project to train a lightweight ResNet model on the CIFAR-10 dataset, achieving strong performance with fewer than 5 million parameters. The project explores two phases: an initial training approach followed by an optimized approach incorporating advanced techniques like cosine annealing, CutMix augmentation, and Test-Time Augmentation (TTA).

## Project Overview

- **Dataset**: CIFAR-10 (10 classes, 50,000 training images, 10,000 test images, 32x32 RGB)
- **Model**: Custom `ResNetSmall` architecture based on ResNet with `BasicBlock`, designed to have fewer than 5 million parameters
- **Goal**: Develop an efficient model with high accuracy on CIFAR-10, then enhance it through optimization for better generalization

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
- **Optimizer**: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **Scheduler**: MultiStepLR (milestones=[50, 75], gamma=0.1)
- **Loss**: Cross-Entropy Loss
- **Epochs**: 100
- **Data Augmentation**: Random crop, horizontal flip, normalization
- **Model Saving**: Best model based on test accuracy saved
- **Results**: Achieved ~94% test accuracy

### Optimized Training
- **Optimizer**: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- **Scheduler**: CosineAnnealingLR (T_max=200)
- **Loss**: Cross-Entropy Loss
- **Epochs**: 200
- **Data Augmentation**: Random crop, horizontal flip, normalization, plus CutMix (alpha=0.2) applied during training to mix inputs and targets
- **Test-Time Augmentation (TTA)**: Predictions averaged over multiple augmented test image versions
- **Results**: Achieved training accuracy of 80--85% and test accuracy of 90--95%, with test accuracy exceeding training accuracy due to CutMix regularization and TTA generalization

## Key Improvements
- **CutMix**: Replaced Mixup in the optimized phase, enhancing regularization by mixing image patches and labels
- **Cosine Annealing**: Provided smoother learning rate decay compared to MultiStepLR
- **Extended Epochs**: Increased to 200 for deeper convergence
- **TTA**: Boosted test performance by leveraging augmentation at inference

## Results
- **Initial Model**: ~94% test accuracy after 100 epochs
- **Optimized Model**: ~90--95% test accuracy after 200 epochs with CutMix and TTA, demonstrating improved generalization

## Contributions
- **Ranjan (sp8171@nyu.edu)**
- **Nishanth (nk3968@nyu.edu)**
- **Navdeep (nm4686@nyu.edu)**
