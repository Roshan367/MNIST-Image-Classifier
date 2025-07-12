# Imports
import torch
import sys
import os
import torchvision
import numpy as np
import joblib
from torchvision.datasets import MNIST

"""
Loads the dataset in the correct format needed
for the different models
"""


def get_dataset(split, model_type="default"):
    batch_size_train = 64
    batch_size_test = 1000

    # loads training and test dataset as tensors for Pytorch CNN model
    if model_type == "pytorch cnn":
        # Training set
        train_loader = torch.utils.data.DataLoader(
            MNIST(
                "/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=batch_size_train,
            shuffle=True,
        )

        # testing set
        test_loader = torch.utils.data.DataLoader(
            MNIST(
                "/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
                train=False,
                download=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                ),
            ),
            batch_size=batch_size_test,
            shuffle=True,
        )
        if split == "train":
            return train_loader
        else:
            return test_loader

    # loads Training and Testing dataset for the custom scratch CNN
    elif model_type == "cnn":
        # training set
        train_dataset = MNIST(
            root="/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
            train=True,
            download=True,
            transform=None,
        )

        # testing set
        test_dataset = MNIST(
            root="/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
            train=False,
            download=True,
            transform=None,
        )

        # gets the training and testing images and labels
        X_train = train_dataset.data.float() / 255.0  # Normalize to [0.0, 1.0]
        y_train = train_dataset.targets  # Labels: shape [60000]
        X_test = test_dataset.data.float() / 255.0
        y_test = test_dataset.targets

        if split == "train":
            # Converts the data in numpy arrays
            return X_train.numpy(), y_train.numpy()
        else:
            return X_test.numpy(), y_test.numpy()

    # loads training and testing dataset for custom PCA + KNN model
    else:
        mnist_train = MNIST(
            root="/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
            train=True,
            download=True,
        )

        mnist_test = MNIST(
            root="/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
            train=False,
            download=True,
        )
        # Convert images to numpy arrays and flatten
        X_train = np.array([np.array(img).reshape(-1) for img, _ in mnist_train])
        y_train = np.array([label for _, label in mnist_train])

        X_test = np.array([np.array(img).reshape(-1) for img, _ in mnist_test])
        y_test = np.array([label for _, label in mnist_test])

        if split == "train":
            return X_train, y_train
        else:
            # only uses first 3000 images in testing to prevent long testng time
            return X_test[:3000], y_test[:3000]


"""
Saves the KNN + PCA model
"""


def save_model(model, filename="models/trained_model.pkl"):
    # Save the model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


"""
Loads the KNN + PCA model
"""


def load_model(filename="models/trained_model.pkl"):
    # Load model from file with error handling if file is missing
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        sys.exit()


"""
Saves the custom CNN model
"""


def save_model_cnn(conv, full, filename="models/cnn_model.pkl"):
    model_data = {"conv": conv.get_params(), "full": full.get_params()}

    with open(filename, "wb") as f:
        joblib.dump(model_data, f)


"""
Loads the custom CNN model
"""


def load_model_cnn(conv, full, filename="models/cnn_model.pkl"):
    with open(filename, "rb") as f:
        model_data = joblib.load(f)

    # Sets the weights and biases for the model
    conv.set_params(model_data["conv"])
    full.set_params(model_data["full"])
