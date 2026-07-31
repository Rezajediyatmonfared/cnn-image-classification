# CIFAR-10 Image Classification with TensorFlow/Keras

This project explores image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) implemented with TensorFlow and Keras. It demonstrates the process from data loading and preprocessing to building, training, evaluating, and visualizing the performance of two distinct CNN architectures: a base CNN and a CNN incorporating dropout layers to mitigate overfitting.

## Dataset

The CIFAR-10 dataset is used, which consists of:
- **60,000** total images.
- **50,000** training images.
- **10,000** testing images.
- Images are **32x32 pixels** with **3 color channels (RGB)**.
- The dataset is divided into **10 mutually exclusive classes**:
    - `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

## Preprocessing

Before feeding the images into the neural network, the following preprocessing steps were applied:
1.  **Pixel Normalization**: Pixel values, originally in the range of 0-255, were scaled to the range of 0.0-1.0 by dividing by 255. This helps in stabilizing the training process.
2.  **One-Hot Encoding**: The integer class labels for the training and testing sets were converted into a one-hot encoded format. For example, a 'dog' (class index 5) would be represented as a vector `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]`. This is necessary for using the `categorical_crossentropy` loss function.

## Models

Two CNN models were implemented and trained:

### 1. Base CNN Model
This is a standard CNN architecture composed of convolutional and pooling layers designed to extract features from images, followed by fully connected layers for classification.
- It utilizes `Conv2D` and `MaxPooling2D` layers.
- A `Flatten` layer converts the 2D feature maps into a 1D vector.
- A `Dense` layer with ReLU activation processes the flattened features.
- The final output layer is a `Dense` layer with a `softmax` activation function to produce probabilities for each of the 10 classes.

### 2. CNN Model with Dropout
This model builds upon the base CNN architecture by introducing dropout layers. Dropout is a regularization technique that randomly sets a fraction of neuron activations to zero during training.
- **Dropout Layers**:
    - A dropout layer with a rate of `0.25` is applied after each pooling layer.
    - A dropout layer with a rate of `0.5` is applied after the dense layer (before the output layer).
- **Benefits**: Dropout helps prevent overfitting by reducing the model's reliance on specific neurons and encouraging it to learn more robust features.
- **Parameters**: This model has approximately 669,354 trainable parameters.

## Training Configuration

Both models were trained using the following hyperparameters:
- **Epochs**: 31
- **Batch Size**: 256
- **Learning Rate**: 0.001
- **Optimizer**: RMSprop
- **Loss Function**: `categorical_crossentropy`
- **Metrics**: `accuracy`
- **Validation Split**: 30% of the training data was used for validation during training.

## Evaluation

The performance of the models was evaluated using:
- **Loss and Accuracy Plots**: Training and validation loss/accuracy were plotted against epochs for both models to visualize learning progress and identify potential overfitting.
- **Model Evaluation on Test Set**: The `evaluate` method was used to compute the final loss and accuracy on the unseen test dataset.
- **Prediction Visualization**: A function was implemented to display a grid of images from the test set, showing the true label versus the model's predicted label. Correct predictions are highlighted in green, and incorrect ones in red. A batch accuracy was also reported.
- **Confusion Matrix**: A confusion matrix was generated for the entire test set to provide a detailed breakdown of classification performance per class, revealing which classes the model tends to confuse.

## Results

*(Note: Specific numerical results like final test accuracy, loss values, and confusion matrix details would typically be populated here based on the actual output of the training runs.)*

The project successfully demonstrates the implementation and comparison of two CNNs for image classification, highlighting the impact of dropout regularization on model performance and generalization.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow (`tensorflow`)
- Keras (included with TensorFlow)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Seaborn (`seaborn`)

### Installation
```bash
pip install tensorflow numpy matplotlib seaborn
