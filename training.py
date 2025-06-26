import numpy as np
from utils import *
from convolution import *
from max_pooling import *
from layers import *
from loss import *
import tensorflow.keras as keras
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.mnist.load_data()
)

X_train = train_images / 255.0
y_train = train_labels

y_train = to_categorical(y_train)

conv = Convolution(X_train[0].shape, 6, 1)
pool = MaxPool(2)
full = Connected(121, 10)


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
    save_model(conv, full)


def predict(input_sample, conv, pool, full):
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)

    flattened_output = pool_out.flatten()

    predictions = full.forward(flattened_output)
    return predictions


def main():
    train_network(X_train, y_train, conv, pool, full)


if __name__ == "__main__":
    main()
