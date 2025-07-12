# Imports
from utils import *
import torch
import os
import PCA_KNN.system as system
import torch.nn.functional as F
import Pytorch_CNN.system_nn as pytorch_cnn
import torch.optim as optim
import numpy as np
import Scratch_CNN.convolution as scratch_conv
import Scratch_CNN.max_pooling as scratch_pool
import Scratch_CNN.layers as scratch_layers
import Scratch_CNN.loss as scratch_loss
from keras.utils import to_categorical

os.makedirs("models", exist_ok=True)

# Loads the datasets for the different models
X_scratch_cnn_train, y_scratch_cnn_train = get_dataset("train", "cnn")
y_scratch_cnn_train = to_categorical(y_scratch_cnn_train)

cnn_train_loader = get_dataset("train", "pytorch cnn")

X_knn_train, y_knn_train = get_dataset("train")

# Setting up custom cnn
conv = scratch_conv.Convolution(X_scratch_cnn_train[0].shape, 6, 1)
pool = scratch_pool.MaxPool(2)
full = scratch_layers.Connected(121, 10)

# setting up pytorch cnn
log_interval = 10
learning_rate = 0.02
momentum = 0.5

network = pytorch_cnn.Net()
optimiser = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []

"""
Trains the Pytorch CNN

Performs the forward and back passes and updates the parameters using batches
"""


def train_cnn(epoch):
    # set model to training mode
    network.train()
    for batch_idx, (data, target) in enumerate(cnn_train_loader):
        # clear previous gradients
        optimiser.zero_grad()
        # forward pass
        output = network(data)
        # compute loss
        loss = F.nll_loss(output, target)
        # backward pass
        loss.backward()
        # update model parameters
        optimiser.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(cnn_train_loader.dataset),
                    100.0 * batch_idx / len(cnn_train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(cnn_train_loader.dataset))
            )


"""
Training for the PCA and KNN model

Performs dimensionality reduction and then performs KNN
"""


def train_knn():
    # Extract dimension-reduced features for training
    train_feature_vectors = system.image_to_reduced_feature(X_knn_train, "train")
    # Train the classifier
    model = system.training_model(train_feature_vectors, y_knn_train)
    # Save the trained model
    save_model(model)


"""
Training for the custom CNN

Performs the forward and backward passes
"""


def train_scratch_cnn(X, y, conv, pool, full, lr=0.01, epochs=3):
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        # iterates for each image
        for i in range(len(X)):
            # forward pass
            conv_out = conv.forward(X[i])
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)
            # computes loss
            loss = scratch_loss.cross_entropy_loss(full_out.flatten(), y[i])
            total_loss += loss

            # converts to a one-hot encoded vector
            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y[i])

            # increments counter for correct predictions
            if num_pred == num_y:
                correct_predictions += 1

            gradient = scratch_loss.cross_entropy_loss_gradient(
                y[i], full_out.flatten()
            ).reshape((-1, 1))
            # backward pass
            full_back = full.backward(gradient, lr)
            pool_back = pool.backward(full_back, lr)
            conv_back = conv.backward(pool_back, lr)

        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_scratch_cnn_train) * 100.0
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%"
        )
    # saves model
    save_model_cnn(conv, full)


def main():
    print("Training Custom CNN Model")
    train_scratch_cnn(X_scratch_cnn_train, y_scratch_cnn_train, conv, pool, full)

    print("Training KNN Model")
    train_knn()

    print("Training CNN Model")
    for epoch in range(1, 3 + 1):
        train_cnn(epoch)
    torch.save(network.state_dict(), "models/cnn_model.pth")


if __name__ == "__main__":
    main()
