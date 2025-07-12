# imports
import tkinter as tk
import torch
import torch.nn.functional as F
import Pytorch_CNN.system_nn as pytorch_cnn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from utils import *
import PCA_KNN.system as system
import Scratch_CNN.loss as scratch_loss
import Scratch_CNN.convolution as conv
import Scratch_CNN.max_pooling as pool
import Scratch_CNN.layers as layers
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

# Loads the trained models
pca_knn_model = load_model()

pytorch_cnn_model = pytorch_cnn.Net()
pytorch_cnn_model.load_state_dict(torch.load("models/cnn_model.pth"))

conv_test = conv.Convolution((28, 28), 6, 1)
pool_test = pool.MaxPool(2)
full_test = layers.Connected(121, 10)
load_model_cnn(conv_test, full_test)

test_losses = []

"""
Class for the GUI for testing the accuracy of the test
datasets and predicting user drawing using the three models
"""


class DigitClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Classifier")

        self.canvas = tk.Canvas(master, width=280, height=280, bg="white")
        self.canvas.place(anchor=tk.NW)

        self.button_predict = tk.Button(
            master, text="Predict Image KNN", command=self.predict
        )
        self.button_predict.place(x=0, y=290)

        self.button_predict_nn = tk.Button(
            master, text="Predict Image CNN", command=self.predict_nn
        )
        self.button_predict_nn.place(x=0, y=330)

        self.button_predict_scratch_nn = tk.Button(
            master, text="Predict Image Custom CNN", command=self.predict_scratch_nn
        )
        self.button_predict_scratch_nn.place(x=0, y=370)

        self.button_clear = tk.Button(master, text="Clear", command=self.clear)
        self.button_clear.place(x=0, y=410)

        self.label_knn = tk.Label(master, text="KNN Prediction: ")
        self.label_knn.place(x=150, y=295)

        self.label_cnn = tk.Label(master, text="CNN Prediction: ")
        self.label_cnn.place(x=150, y=335)

        self.label_scratch_cnn = tk.Label(master, text="Custom CNN Prediction: ")
        self.label_scratch_cnn.place(x=200, y=375)

        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.button_test = tk.Button(
            master, text="Test Accuracy KNN", command=self.test
        )
        self.button_test.place(x=350, y=30)

        self.button_test_nn = tk.Button(
            master, text="Test Accuracy CNN", command=self.test_nn
        )
        self.button_test_nn.place(x=350, y=70)

        self.button_test_scratch_nn = tk.Button(
            master, text="Test Accuracy Scratch CNN", command=self.test_scratch_nn
        )
        self.button_test_scratch_nn.place(x=350, y=110)

        self.label_knn_accuracy = tk.Label(master, text="KNN Accuracy (Clean): ")
        self.label_knn_accuracy.place(x=500, y=35)

        self.label_cnn_accuracy = tk.Label(master, text="CNN Accuracy (Clean): ")
        self.label_cnn_accuracy.place(x=500, y=75)

        self.label_cnn_scratch_accuracy = tk.Label(
            master, text="CNN Scratch Accuracy (Clean): "
        )
        self.label_cnn_scratch_accuracy.place(x=550, y=115)

    """
    Allows user to draw on the canvas
    """

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=8)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    """
    Clears the canvas of current drawings
    """

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")
        self.label_knn.config(text="KNN Prediction: ")
        self.label_cnn.config(text="CNN Prediction: ")

    """
    Performs PCA+KNN prediction of the canvas image
    using the loaded trained model
    """

    def predict(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img).reshape(1, -1)  # Flatten to 1x784

        # PCA reduction
        pca_input = system.image_to_reduced_feature(img_array)

        # Predict
        prediction = pca_knn_model.predict(pca_input)
        predicted_digit = prediction[0]
        text = f"KNN Prediction: {predicted_digit}"

        self.label_knn.config(text=text)

    """
    Performs Pytorch CNN prediction of the canvas image using
    the loaded trained model
    """

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

        pytorch_cnn_model.eval()
        with torch.no_grad():
            output = pytorch_cnn_model(img_tensor)
            predicted_digit = output.argmax(dim=1, keepdim=True).item()
        text = f"CNN Prediction: {predicted_digit}"
        self.label_cnn.config(text=text)

    def predict_scratch_nn(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img_array = np.array(img)

        pred = scratch_loss.predict(img_array, conv_test, pool_test, full_test)
        text = f"Custom CNN Prediction: {np.argmax(pred)}"
        self.label_scratch_cnn.config(text=text)

    """
    Tests the PCA+KNN model on a limited clean testing set using
    the loaded trained model
    """

    def test(self):
        X_knn_test, y_knn_test = get_dataset("test")
        test_feature_vectors = system.image_to_reduced_feature(X_knn_test)
        test_predictions = pca_knn_model.predict(test_feature_vectors)
        test_accuracy = accuracy_score(y_knn_test, test_predictions) * 100
        text = f"KNN Accuracy (Clean): {round(test_accuracy, 3)}%"
        self.label_knn_accuracy.config(text=text)

    """
    Tests the Pytorch CNN model on a clean testing set using the 
    loaded trained model
    """

    def test_nn(self):
        cnn_test_loader = get_dataset("test", "pytorch cnn")

        pytorch_cnn_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in cnn_test_loader:
                output = pytorch_cnn_model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(cnn_test_loader.dataset)
        test_losses.append(test_loss)
        accuracy = 100.0 * correct / len(cnn_test_loader.dataset)
        text = f"CNN Accuracy (Clean): {round(float(accuracy), 3)}%"
        self.label_cnn_accuracy.config(text=text)

    """
    Tests the Custom CNN model on a clean testing set using the loaded
    trained model
    """

    def test_scratch_nn(self):
        X_scratch_cnn_test, y_scratch_cnn_test = get_dataset("test", "cnn")
        y_scratch_cnn_test = to_categorical(y_scratch_cnn_test)

        predictions = []

        for data in X_scratch_cnn_test:
            pred = scratch_loss.predict(data, conv_test, pool_test, full_test)
            one_hot_pred = np.zeros_like(pred)
            one_hot_pred[np.argmax(pred)] = 1
            predictions.append(one_hot_pred.flatten())

        predictions = np.array(predictions)
        accuracy = accuracy_score(predictions, y_scratch_cnn_test) * 100.0
        text = f"CNN Scratch Accuracy (Clean): {round(float(accuracy), 3)}%"
        self.label_cnn_scratch_accuracy.config(text=text)


# runs the GUI
root = tk.Tk()
app = DigitClassifierApp(root)
root.mainloop()
