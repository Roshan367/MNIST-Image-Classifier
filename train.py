from utils import *
import torch
import system
import torch.nn.functional as F
from system_nn import *
import torch.optim as optim
import numpy as np
from convolution import *
from max_pooling import *
from layers import *
from loss import *
from keras.utils import to_categorical


X_train, y_train = get_dataset("train", tensor="cnn")
y_train = to_categorical(y_train)

conv = Convolution(X_train[0].shape, 6, 1)
pool = MaxPool(2)
full = Connected(121, 10)

# training pytorch cnn
log_interval = 10
learning_rate = 0.02
momentum = 0.5

network = Net()
optimiser = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

train_loader = get_dataset("train", tensor="pytorch cnn")

train_losses = []
train_counter = []


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimiser.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimiser.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )


def train_network(X, y, conv, pool, full, lr=0.01, epochs=5):
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        for i in range(len(X)):
            conv_out = conv.forward(X[i])
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)
            loss = cross_entropy_loss(full_out.flatten(), y[i])
            total_loss += loss

            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y[i])

            if num_pred == num_y:
                correct_predictions += 1

            gradient = cross_entropy_loss_gradient(y[i], full_out.flatten()).reshape(
                (-1, 1)
            )
            full_back = full.backward(gradient, lr)
            pool_back = pool.backward(full_back, lr)
            conv_back = conv.backward(pool_back, lr)

        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_train) * 100.0
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%"
        )
    save_model_cnn(conv, full)


def predict(input_sample, conv, pool, full):
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)

    flattened_output = pool_out.flatten()

    predictions = full.forward(flattened_output)
    return predictions


def main():
    print("Training Scratch CNN")
    train_network(X_train, y_train, conv, pool, full)
    print("Training KNN Model")
    # Load training data
    train_images, train_labels = get_dataset("train")

    # Extract dimension-reduced features for training
    train_feature_vectors = system.image_to_reduced_feature(train_images, "train")

    # Train the classifier
    model = system.training_model(train_feature_vectors, train_labels)

    # Save the trained model
    save_model(model)

    train_loader = get_dataset("train", tensor=True)

    print("Training CNN Model")
    for epoch in range(1, n_epochs + 1):
        train(epoch)
    torch.save(network.state_dict(), "cnn_model.pth")


if __name__ == "__main__":
    main()
