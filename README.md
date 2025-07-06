# MNIST-Image Classifier

## Project Summary

This project is a MNIST-image classifer made up of three different models, which is trained on handwritten images of numbers (0-9). A GUI interface can then be used to test the models on test datasets or predict user input.

## Setup
- Libraries Used:
    - numpy
    - sklearn
    - scipy
    - torch
    - torchvision
    - keras
    - tkinter
    - PIL
    - joblib

Clone the repository
```
git clone https://github.com/Roshan367/MNIST-Image-Classifier.git
```
Go to the directory
```
cd MNIST-Image-Classifier/
```
Train models
```
python train.py
```
Run GUI interface
```
python gui_classification.py
```

## Models

### Custom PCA+KNN Model

One of the models used is a hand-made Principal Component Analysis and K-Nearest Neighbour model with optimised hyperparameters.

The Principal Component Analysis is used for dimensionality reduction of the images and the K-Nearest Neighbour is used for classification.

The optimisation of the component number, for PCA, was done by plotting an explained variance graph and 

The optimisation of the k number, for KNN, was done through using cross-validation of a random subset of the training data, that was randomly sorted.

Average Accuracy - 90-95%

### PyTorch CNN

The next model is a Convolutional Neural Network using PyTorch.

It uses two convolutional, max-pooling and fully connected layers. Dropout is also used for the second convolutional network and for the first fully connected network, to prevent overfitting.

Batches are also used during training

Activation function(s) - ReLU, Log-Softmax

Loss Function - Negative Log-Liklihood

Average Accurracy (3 Epochs) - 95-98%

### Custom CNN

The final model is a hand-made Convolutional Neural Network utilising only numpy and scipy.

It uses a single convolutional, max-pooling and fully connected layer.

Activation function(s) - Softmax

Loss Function - Cross-Entropy

Average Accuracy (5 Epochs) - 85-88%

