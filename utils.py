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


def get_dataset(split, tensor="default"):
    batch_size_train = 64
    batch_size_test = 1000
    if tensor == "pytorch cnn":
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

    elif tensor == "cnn":
        train_dataset = MNIST(
            root="/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
            train=True,
            download=True,
            transform=None,
        )
        test_dataset = MNIST(
            root="/home/roshan/Documents/MNIST-Image-Classifier/mnist_subset/",
            train=False,
            download=True,
            transform=None,
        )
        X_train = train_dataset.data.float() / 255.0  # Normalize to [0.0, 1.0]
        y_train = train_dataset.targets  # Labels: shape [60000]
        X_test = test_dataset.data.float() / 255.0
        y_test = test_dataset.targets

        if split == "train":
            return X_train.numpy(), y_train.numpy()
        else:
            return X_test.numpy(), y_test.numpy()

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


def save_model(model, filename="models/trained_model.pkl"):
    # Save the model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename="models/trained_model.pkl"):
    # Load model from file with error handling if file is missing
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        sys.exit()


def save_model_cnn(conv, full, filename="models/cnn_model.pkl"):
    model_data = {"conv": conv.get_params(), "full": full.get_params()}

    with open(filename, "wb") as f:
        joblib.dump(model_data, f)


def load_model_cnn(conv, full, filename="models/cnn_model.pkl"):
    with open(filename, "rb") as f:
        model_data = joblib.load(f)

    conv.set_params(model_data["conv"])
    full.set_params(model_data["full"])
