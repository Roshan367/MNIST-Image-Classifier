# imports
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# sets variables used in model
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.02
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

"""
Pytorch CNN class for the image classification

Applies two convolutional layers with max pooling,
ReLU activation and two fully connected layers

Dropout for regularization preventing overfitting
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # applies dropout after second convolutional layer
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # dropout applied after first fully connected layer
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    main()
