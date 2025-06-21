from utils import *
import torch
import system
import torch.nn.functional as F
from system_nn import *
import torch.optim as optim

log_interval = 10
learning_rate = 0.02
momentum = 0.5


network = Net()
optimiser = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

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


def main():
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
