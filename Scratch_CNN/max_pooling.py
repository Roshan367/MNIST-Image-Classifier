# imports
import numpy as np

"""
Max Pooling layer for the CNN

Applies pooling to reduce the spatial dimensions
of the feature maps and contains functionality of
the forward and backward passes
"""


class MaxPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    """
    Performs the forward pass for the max pooling layer

    Reduces the each channel by sliding over the filter
    """

    def forward(self, input_data):
        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        self.output = np.zeros(
            (self.num_channels, self.output_height, self.output_width)
        )
        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    self.output[c, i, j] = np.max(patch)

        return self.output

    """
    Performs the backward pass of the max pooling layer

    Gets the patch and creates a boolean mask which is
    True for only the max values in the patch
    
    The mask is then used to route the gradient
    """

    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)

                    dL_dinput[c, start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask

        return dL_dinput
