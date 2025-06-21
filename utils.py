import pandas as pd
import torch
import sys
import os
from PIL import Image
import torchvision
from torchvision import transforms
import numpy as np
import joblib
from torchvision.datasets import MNIST


def get_dataset(split, tensor=False):
    batch_size_train = 64
    batch_size_test = 1000
    if tensor:
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
    else:
        # Load raw MNIST PIL images
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
        X_train = np.array(
            [np.array(img).reshape(-1) for img, _ in mnist_train]
        )  # shape: (N, 784)
        y_train = np.array([label for _, label in mnist_train])

        X_test = np.array([np.array(img).reshape(-1) for img, _ in mnist_test])
        y_test = np.array([label for _, label in mnist_test])

        if split == "train":
            return X_train, y_train
        else:
            return X_test[:3000], y_test[:3000]


def save_model(model, filename="trained_model.pkl"):
    # Save the model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename="trained_model.pkl"):
    # Load model from file with error handling if file is missing
    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"File {filename} not found")
        sys.exit()
