# Intel Image Classification

A complete image analysis and classification project based on the Intel Image Classification dataset.

This project explores image preprocessing, RGB channel analysis, statistical analysis, classical machine learning regression, artificial neural networks, convolutional neural networks, binary classification, ROC curves, AUC evaluation, and stratified cross-validation.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dataset Classes](#dataset-classes)
- [Project Objectives](#project-objectives)
- [Technologies and Libraries](#technologies-and-libraries)
- [Project Workflow](#project-workflow)
- [Data Analysis](#data-analysis)
- [Image Preprocessing](#image-preprocessing)
- [Brightness Prediction](#brightness-prediction)
- [Grayscale Prediction Using an ANN](#grayscale-prediction-using-an-ann)
- [Multiclass CNN Classification](#multiclass-cnn-classification)
- [CNN Evaluation](#cnn-evaluation)
- [Stratified Cross-Validation](#stratified-cross-validation)
- [Binary Classification](#binary-classification)
- [ROC Curve and AUC](#roc-curve-and-auc)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## Project Overview

The purpose of this project is to analyze and classify natural landscape images using different data science and deep learning techniques.

The project starts with basic dataset exploration and gradually develops more advanced models. The main classification task is performed using a Convolutional Neural Network (CNN), while additional experiments are conducted using linear regression, polynomial regression, artificial neural networks, binary classification, and cross-validation.

The project demonstrates a complete machine learning workflow:

1. Dataset inspection
2. Image loading
3. Image preprocessing
4. Statistical analysis
5. Visualization
6. Regression experiments
7. Neural network modeling
8. CNN-based image classification
9. Model evaluation
10. Cross-validation
11. Binary classification analysis

---

## Dataset

This project uses the Intel Image Classification dataset.

The dataset contains natural landscape images divided into six categories. Each image is an RGB image representing a different type of environment or landscape.

The original dataset is available on Kaggle:

[Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

### Dataset Path

The dataset path used during development is:
```python
dataset_path = r"C:\Users\Ehadish\seg_train\seg_train"

