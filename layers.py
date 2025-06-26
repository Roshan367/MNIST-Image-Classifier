import numpy as np


class Connected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.randn(output_size, 1)

    def softmax(self, z):
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)
        log_sum_exp = np.log(sum_exp_values)

        probabilities = exp_values / sum_exp_values

        return probabilities

    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)

    def forward(self, input_data):
        self.input_data = input_data
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases

        self.output = self.softmax(self.z)
        return self.output

    def backward(self, dL_dout, lr):
        dL_dy = np.dot(self.softmax_derivative(self.output), dL_dout)

        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))

        dL_db = dL_dy

        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db

        return dL_dinput

    def get_params(self):
        return {"weights": self.weights, "biases": self.biases}

    def set_params(self, params):
        self.weights = params["weights"]
        self.biases = params["biases"]
