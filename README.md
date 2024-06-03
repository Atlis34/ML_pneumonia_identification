# ML_pneumonia_identification
Project #4
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

# Pneumonia Detection with Deep Learning

This project aims to develop a deep learning model to identify pneumonia from chest X-ray images. Using the Xception architecture and transfer learning, we achieve high accuracy and reliability, making this model a potential tool for assisting radiologists.

## Table of Contents
- [Overview](#overview)
- [Model Summary](#model-summary)
- [Evaluation Metrics](#evaluation-metrics)
- [Correct and Incorrect Predictions](#correct-and-incorrect-predictions)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

Pneumonia is a serious lung infection that requires timely and accurate diagnosis. This project uses a convolutional neural network (CNN) to automatically detect pneumonia from chest X-ray images, potentially aiding healthcare professionals in diagnosis.

## Model Summary

We used the Xception architecture with transfer learning to leverage pre-trained weights from the ImageNet dataset. The model includes global average pooling and dense layers to adapt it for binary classification (pneumonia vs. no pneumonia).

- **Xception Model**: Pre-trained on ImageNet
- **Global Average Pooling**: Reduces the spatial dimensions of the feature maps
- **Dense Layers**: Custom layers for binary classification
- **Dropout**: Regularization to prevent overfitting

## Evaluation Metrics

The model was evaluated using several key metrics:
- **Precision**: 0.86
- **Recall**: 0.95
- **F1 Score**: 0.90
- **Accuracy**: 0.87

The confusion matrix shows the distribution of true positives, true negatives, false positives, and false negatives.

## Setup and Installation
### Prerequisites

- Python 3.x
- TensorFlow 2.x
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook (optional, for running and visualizing the notebook)

## Authors and Acknowlegdgments
 - Katreece Hattaway
 - Ryan MacFarlane
 - Sophia Liu
 - Nicole Anderson

 - Chat GPT, Xpert Learning Assistant, kaggle.com and support from UT Data Anlaytics Staff

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/pneumonia-detection.git
