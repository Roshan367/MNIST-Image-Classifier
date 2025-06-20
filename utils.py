import pandas as pd
import torch
import sys
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import joblib


def get_dataset(split, tensor=False):
    # Load dataset based on the split type
    csv_file_path = "./mnist_subset/image_labels.csv"
    df = pd.read_csv(csv_file_path)
    data_df = df[df["Split"] == split]

    if split == "train":
        data_images_dir = "./mnist_subset/images/"
    elif split == "noise_test":
        data_images_dir = "./mnist_subset/noisy_images/"
    elif split == "mask_test":
        data_images_dir = "./mnist_subset/masked_images/"
    else:
        print("Error")
        sys.exit()

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    data_images = []
    data_labels = []

    for _, row in data_df.iterrows():
        image_path = os.path.join(data_images_dir, row["Filename"])
        img = Image.open(image_path)
        if tensor:
            img = transform(img)
        else:
            img = np.array(img)
        data_images.append(img)
        data_labels.append(int(row["Label"]))

    if tensor:
        data_images = torch.stack(data_images)
        data_labels = torch.tensor(data_labels, dtype=torch.long)
        return data_images, data_labels
    else:
        return np.array(data_images), np.array(data_labels)


def save_model(model, filename="trained_model.pkl"):
    # Save the model to a file
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename="trained_model.pkl"):
    # Load model from file with error handling if file is missing
    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"File {filename} not found")
        sys.exit()
