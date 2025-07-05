# imports
import numpy as np

"""
Loss function used at the end of the forward pass
"""


def cross_entropy_loss(predictions, targets):
    num_samples = 10

    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss


"""
Derivative of the loss function used for the backward pass
"""


def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples

    return gradient


"""
Prediction function used after the model has been trained
"""


def predict(input_sample, conv, pool, full):
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)

    flattened_output = pool_out.flatten()

    predictions = full.forward(flattened_output)
    return predictions
