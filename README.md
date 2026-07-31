# Intel Image Analysis and Classification

This project explores the **Intel Image Classification dataset** through a combination of:

- **Statistical image analysis**
- **Pixel-level regression modeling**
- **Artificial Neural Network (ANN) regression**
- **Convolutional Neural Network (CNN) classification**
- **Cross-validation**
- **Binary classification with ROC-AUC evaluation**

The goal of this project is not only to classify images, but also to analyze image data statistically and investigate how classical machine learning and deep learning methods can be applied to visual data.

---

## Dataset

The dataset contains natural scene images from the following categories:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

Images are loaded from the Intel scene classification dataset and resized to **150 × 150** pixels for model training and analysis.

---

## Project Objectives

This project was designed to cover multiple stages of image-based data analysis:

1. Load and inspect class distributions
2. Perform pixel-level statistical analysis on RGB channels
3. Visualize distributions using histograms
4. Check normality of channel intensity values
5. Apply linear and polynomial regression on pixel brightness
6. Predict grayscale values using an ANN regressor
7. Normalize image data
8. Train a CNN for multi-class image classification
9. Evaluate classification performance using a confusion matrix and classification report
10. Apply stratified cross-validation
11. Build a binary classifier and evaluate it with ROC curve and AUC score

---

## Workflow

### 1. Data Loading
- Images are loaded class by class
- Each image is resized to `150x150`
- RGB conversion is applied using OpenCV
- Labels are assigned numerically

### 2. Statistical Analysis
The RGB channels are separated and analyzed using:

- Mean
- Median
- Mode
- Standard deviation
- Quartiles of class counts

This helps understand the general intensity distribution of the dataset.

### 3. Histogram Visualization
Histograms are plotted for the Red, Green, and Blue channels to visually inspect the frequency of pixel intensities.

### 4. Normality Test
A statistical normality test is applied to sampled channel values using `scipy.stats.normaltest`.

### 5. Linear Regression
A linear regression model is used to predict **brightness** from the **Red channel**:

- Input: Red channel intensity
- Target: Average brightness of RGB channels

Evaluation metrics:
- $R^2$
- Mean Squared Error (MSE)

### 6. Polynomial Regression
A degree-2 polynomial regression model is trained to compare performance against linear regression.

### 7. ANN Regression
A multi-layer perceptron regressor (`MLPRegressor`) is used to predict grayscale intensity from RGB values:

- Input: `(R, G, B)`
- Output: grayscale value

This demonstrates a nonlinear regression approach on pixel-level data.

### 8. CNN Classification
A Convolutional Neural Network is trained for 6-class image classification using normalized image data.

Architecture includes:
- Convolution layers
- MaxPooling
- Flatten
- Dense layers
- Dropout regularization
- Softmax output layer

### 9. Performance Evaluation
The classification model is evaluated using:

- Validation accuracy and loss curves
- Confusion matrix
- Classification report (precision, recall, F1-score)

### 10. Stratified K-Fold Cross Validation
To assess model robustness, `StratifiedKFold` is applied to the dataset and CNN accuracy is recorded across folds.

### 11. Binary Classification + ROC Curve
A separate binary classification experiment is conducted using two classes:

- `forest`
- `mountain`

The binary CNN model is evaluated using:

- ROC Curve
- AUC Score

---

## Technologies and Libraries

This project uses the following Python libraries:

- `NumPy`
- `Matplotlib`
- `Seaborn`
- `SciPy`
- `OpenCV`
- `Scikit-learn`
- `TensorFlow / Keras`

---

## Project Structure
```text
intel-image-analysis-classification/
├── notebook/
│   └── intel_image_analysis_classification.ipynb
├── images/
│   ├── rgb_histograms.png
│   ├── cnn_accuracy_loss.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
└── README.md
