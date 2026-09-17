# 🚦 Traffic Sign Recognition Using CNN

## Project Overview

Traffic Sign Recognition is a deep learning project that uses a Convolutional Neural Network (CNN) to classify traffic sign images into 43 different classes.

The project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Objective

The main objective is to automatically recognize traffic signs from images and predict the corresponding traffic sign class.

## Technologies Used

- Python
- TensorFlow
- Keras
- CNN
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## CNN Architecture

The model consists of:

- Conv2D
- Batch Normalization
- MaxPooling2D
- Conv2D
- Batch Normalization
- MaxPooling2D
- Conv2D
- Batch Normalization
- MaxPooling2D
- Flatten
- Dense
- Dropout
- Softmax Output Layer

## Project Workflow

Dataset
→ Image Preprocessing
→ Train/Validation Split
→ CNN Model
→ Model Training
→ Model Evaluation
→ Prediction
→ Streamlit Deployment

## How to Run

### 1. Create virtual environment

```bash
python -m venv venv