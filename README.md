# face_detection
FACE DETECTION APP
# Face Recognition with Custom Siamese Network

This repository contains a face recognition model built from scratch using a Siamese neural network. The model leverages a custom architecture to generate embeddings, compute similarity scores, and identify faces based on specified similarity thresholds. This project avoids pre-built face recognition libraries, opting instead for a fully customized approach using TensorFlow and Keras.

## Project Overview

- **Model Architecture**: Siamese network with Conv2D and Dense layers.
- **Embedding Generation**: Custom embedding layers for each input image.
- **Similarity Measurement**: L1 distance layer to compute similarity between embeddings.
- **Optimization**: Binary Cross-Entropy loss for training with accurate similarity-based prediction.
- **Accuracy**: Validation using threshold-based similarity and accuracy metrics.

## Directory Structure

- `model/`: Contains the custom Siamese model and associated layers.
- `data/`: Includes training and validation images (add your dataset here).
- `notebooks/`: Jupyter notebooks for training and testing the model.
- `utils/`: Helper functions for preprocessing, data loading, and evaluation.

## Getting Started

### Prerequisites

-tensorflow==2.10.0
-keras==2.10.0
-numpy==1.23.0
-pandas==1.5.0
-scikit-learn==1.1.2
-matplotlib==3.5.2
-seaborn==0.11.2
-opencv-python==4.6.0.66


To install the required packages, use:

bash
pip install -r requirements.txt

# Face Recognition Using Siamese Network

This project implements a face recognition system using a custom Siamese neural network built from scratch, without relying on external face recognition libraries. The Siamese network is designed to identify similar faces based on pairwise image comparisons, employing a similarity threshold to evaluate accuracy.

## Dataset
- **Data Collection**: Collected 300 positive images (same identity) and 300 anchor images (reference images for each identity).
- **Negative Samples**: Downloaded a set of random negative images (different identities) to train the model with a balanced dataset.
- **Organization**: Arrange data in pairs for training and testing, with each pair consisting of either similar or dissimilar images.

## Model Training
1. **Load Data**: Use `utils/data_loader.py` to load and preprocess images.
2. **Train Model**: Run `notebooks/train_siamese.ipynb` to train the Siamese model.
3. **Evaluate Model**: Calculate accuracy using a custom similarity threshold method.

## Key Functions
- **L1Dist**: Custom layer to compute L1 distance between embeddings, enabling similarity-based comparison.
- **BinaryCrossEntropy**: Loss function used to optimize similarity predictions.
- **Accuracy Calculation**: Custom function that calculates model accuracy in percentage based on similarity thresholds.

## Model Summary
The Siamese network architecture includes Conv2D and Dense layers optimized to generate embeddings. These embeddings are compared through the `L1Dist` layer, which calculates the distance between input image pairs. The model uses binary cross-entropy as the loss function to optimize predictions and enhance accuracy.

## Results
The model achieved high precision in identifying similar faces, as verified through custom accuracy metrics based on similarity thresholds.

## Usage
Use the code snippet below to make predictions with your trained model.

```python
# Example prediction
model.predict([input_image, validation_image])
