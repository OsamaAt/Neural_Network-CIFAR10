# CIFAR-10 Image Classification Using TensorFlow and Keras

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images across 10 classes.

---

## Overview

The goal of this project is to develop a deep learning model capable of accurately classifying images into their respective categories. This project demonstrates the complete workflow, including data preprocessing, model building, training, evaluation, and visualization.

---

## Features

- **Dataset**: CIFAR-10 dataset (`keras.datasets.cifar10`) with 50,000 training images and 10,000 test images.
- **Model Architecture**:
  - Two convolutional layers (32 and 64 filters).
  - MaxPooling, Dropout, Flatten layers.
  - Dense layers for classification using ReLU and softmax activations.
- **Training**: Model trained for 25 epochs with a batch size of 64 and 20% validation split.
- **Evaluation**:
  - Accuracy and loss metrics on test data.
  - Visualization of predictions and actual values.

---

## Installation

Make sure the following dependencies are installed:
- Python (3.x)
- TensorFlow
- NumPy
- Matplotlib

You can install them using pip:
```bash
pip install tensorflow numpy matplotlib
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/CIFAR10-Classifier.git
   cd CIFAR10-Classifier
   ```

2. Execute the script:
   ```bash
   python cifar10_classifier.py
   ```

3. The script will:
   - Preprocess the CIFAR-10 dataset.
   - Train the CNN model.
   - Evaluate the model on test data.
   - Display sample predictions and actual values.

---

## Results

- **Training Accuracy**: ~85%
- **Testing Accuracy**: ~80%
- The model demonstrates effective classification but could benefit from additional tuning.

---

## Key Files

- **cifar10_classifier.py**: The main Python script containing the implementation of the CNN model.
- **README.md**: Project documentation (this file).

---

## Future Enhancements

- **Hyperparameter Tuning**: Optimize parameters like dropout rate, learning rate, and batch size.
- **Additional Layers**: Experiment with more advanced architectures like ResNet or adding BatchNormalization layers.
- **Data Augmentation**: Include techniques like flipping, rotating, and cropping images to improve model robustness.

Author OsamaAt
