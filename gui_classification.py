import tkinter as tk
import torch
import torch.nn.functional as F
from system_nn import *
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from utils import *
import system
from sklearn.metrics import accuracy_score

model = load_model()

test_losses = []


class DigitClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Classifier")

        self.canvas = tk.Canvas(master, width=280, height=280, bg="white")
        self.canvas.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack()

        self.button_predict_nn = tk.Button(
            master, text="Predict NN", command=self.predict_nn
        )
        self.button_predict_nn.pack()

        self.button_test = tk.Button(master, text="Predict", command=self.test)
        self.button_test.pack()

        self.button_test_nn = tk.Button(master, text="Predict", command=self.test_nn)
        self.button_test_nn.pack()

        self.button_clear = tk.Button(master, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label = tk.Label(master, text="Draw a digit and click Predict")
        self.label.pack()

        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")
        self.label.config(text="Draw a digit and click Predict")

    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1, -1)  # Flatten to 1x784

        # PCA reduction
        pca_input = system.image_to_reduced_feature(img_array)

        # Predict
        model = load_model()
        prediction = model.predict(pca_input)
        predicted_digit = prediction[0]

        self.label.config(text=f"Predicted Digit: {predicted_digit}")

    def predict_nn(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # Ensure single channel
                transforms.ToTensor(),  # Convert to tensor [0,1]
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # Use same normalization as training
            ]
        )

        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        model = Net()
        model.load_state_dict(torch.load("cnn_model.pth"))
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
        self.label.config(text=f"Predicted Digit: {pred}")

    def test(self):
        model = load_model()
        noise_test_images, noise_test_labels = get_dataset("test")
        noise_test_feature_vectors = system.image_to_reduced_feature(noise_test_images)
        noise_test_predictions = model.predict(noise_test_feature_vectors)
        noise_test_accuracy = accuracy_score(noise_test_labels, noise_test_predictions)
        self.label.config(text=f"Test Score:{noise_test_accuracy}")

    def test_nn(self):
        test_loader = get_dataset("test", tensor=True)
        model = Net()
        model.load_state_dict(torch.load("cnn_model.pth"))
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        self.label.config(text=f"Accuracy (Clean):{accuracy}")


# Run the app
root = tk.Tk()
app = DigitClassifierApp(root)
root.mainloop()
