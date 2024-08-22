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
git clone https://github.com/smgandhi-18/Brain-Tumor-Detection-CNN
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

## Model Architecture
The model is a Convolutional Neural Network (CNN) consisting of multiple convolutional and pooling layers followed by fully connected layers. Batch normalization and dropout are used to improve training stability and prevent overfitting.

**Layers:**
1. Convolutional Layers: Extract features from the images using filters.
2. MaxPooling Layers: Reduce the spatial dimensions of the feature maps.
3. BatchNormalization: Normalize activations to improve convergence.
4. Dropout: Randomly drop units to prevent overfitting.
5. Fully Connected Layers: Perform the classification task.

**Model Summary:**
1. Input: 128x128 grayscale images
2. Output: Softmax activated layer with 2 output classes

## Training
The model is trained on the training dataset with data augmentation techniques to increase the diversity of the training data. The training is carried out for 30 epochs with a dynamic learning rate adjustment using the LearningRateScheduler.

## Training Procedure:
1. Data Augmentation: Rotation, zoom, width/height shift, shear, and horizontal flip.
2. Optimizer: Adam optimizer with an initial learning rate of 0.001.
3. Loss Function: Sparse Categorical Crossentropy.

## Run the Training:
To train the model, use:
```bash
python train.py
```

## Evaluation
The model is evaluated on a separate test dataset to measure its performance. Various metrics such as accuracy, confusion matrix, and ROC-AUC are used to assess the model.

## Evaluation Metrics:
1. Accuracy: Measures the proportion of correct predictions.
2. Confusion Matrix: Provides insights into the classification performance.
3. ROC-AUC: Measures the trade-off between true positive and false positive rates.

## Run the Evaluation:
```bash
python evaluate.py
```

## Results
The trained model achieves an accuracy of approximately XX% on the test dataset. The confusion matrix and classification report provide detailed performance metrics. The ROC curve and AUC score further highlight the model's ability to distinguish between the two classes.

## License
This project is licensed under the Apache License 2.0.
