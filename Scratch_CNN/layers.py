# imports
import numpy as np

"""
Fully connected layers class for the CNN

Contains the functionality for the forward pass,
backward pass and the activation function
"""


class Connected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # randomly sets weights and biases on initialisation
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.randn(output_size, 1)

    """
    Activation function used after the input input data
    has been processed through the network
    """

    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)
        log_sum_exp = np.log(sum_exp_values)

        probabilities = exp_values / sum_exp_values

        return probabilities

    """
    Performs the forward pass for the fully connected layer

    Flattens the input and performs a linear transformation
    The activation function is then applied to the new matrix
    """

    def forward(self, input_data):
        self.input_data = input_data
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases

        self.output = self.softmax(self.z)
        return self.output

    """
    Performs the backward pass for the fully connected layer

    Calculates the partial differentials and performs gradient
    descent for the weights and biases using the learning rate
    """

    def backward(self, dL_dout, lr):
        dL_dy = dL_dout
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))
        dL_db = dL_dy

        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db

        return dL_dinput

    """
    Gets current weight and bias values
    """

    def get_params(self):
        return {"weights": self.weights, "biases": self.biases}

    """
    Sets filter and bias values
    """

    def set_params(self, params):
        self.weights = params["weights"]
        self.biases = params["biases"]
