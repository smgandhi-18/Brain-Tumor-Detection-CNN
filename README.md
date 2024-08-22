# Brain Tumor Detection Using Convolutional Neural Networks

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Introduction
This project is focused on the detection of brain tumors using Convolutional Neural Networks (CNNs). Brain tumors are abnormal growths of cells in the brain that can be benign or malignant. Early detection of brain tumors is crucial for treatment and improving patient outcomes. This project utilizes deep learning techniques to automate the classification of brain MRI images into categories: "no tumor" and "tumor". The aim is to build a robust model that can assist radiologists in diagnosing brain tumors more effectively.

## Installation
To get started with this project, you need to have Python 3.x installed on your machine along with several libraries.

### Clone the Repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

Dependencies:
1. TensorFlow
2. NumPy
3. OpenCV
4. Matplotlib
5. Scikit-learn
Make sure you have a CUDA-compatible GPU if you intend to train the model on GPU.

## Datasets
The dataset used in this project consists of brain MRI images categorized into two classes: "no tumor" and "tumor". The images are preprocessed into grayscale and resized to 128x128 pixels to reduce computational complexity while retaining essential features.

The dataset can be found at Kaggle: Brain Tumor MRI Dataset (https://www.kaggle.com/datasets/luluw8071/brain-tumor-mri-datasets). Download and extract it to the datasets/ directory in the project folder.
