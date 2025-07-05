# imports
import numpy as np
from scipy.signal import correlate2d

"""
Convolutional layer class for the CNN

Applies a filter to the image for feature extraction
and contains functionality for the forward and backward passes
"""


class Convolution:
    def __init__(self, input_shape, filter_size, num_filters):
        input_height, input_weight = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters

        self.filter_shape = (num_filters, filter_size, filter_size)
        self.output_shape = (
            num_filters,
            input_height - filter_size + 1,
            input_weight - filter_size + 1,
        )

        # sets filters and biases randomly on initialisation
        self.filters = np.random.rand(*self.filter_shape)
        self.biases = np.random.rand(*self.output_shape)

    """
    Performs the forward pass for the convolutional layer

    Cross-correlates the input data with each of the filters
    """

    def forward(self, input_data):
        self.input_data = input_data

        output = np.zeros(self.output_shape)
        for i in range(self.num_filters):
            output[i] = correlate2d(self.input_data, self.filters[i], mode="valid")
        # normalises the output so there are no negative values
        output = np.maximum(output, 0)
        return output

    """
    Performs the backward pass for the convolutional layer

    Calculates partial differentials and performs gradient
    descent for filters and biases using learning rate
    """

    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)
        dL_dfilters = np.zeros_like(self.filters)

        for i in range(self.num_filters):
            dL_dfilters[i] = correlate2d(self.input_data, dL_dout[i], mode="valid")

            dL_dinput += correlate2d(dL_dout[i], self.filters[i], mode="full")

        self.filters -= lr * dL_dfilters
        self.biases -= lr * dL_dout

        return dL_dinput

    """
    Gets current filter and biase values
    """

    def get_params(self):
        return {"filters": self.filters, "biases": self.biases}

    """
    Sets filter and biase values
    """

    def set_params(self, params):
        self.filters = params["filters"]
        self.biases = params["biases"]
