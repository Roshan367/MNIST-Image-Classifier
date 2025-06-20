import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from utils import *
import system

model = load_model()


class DigitClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("MNIST Digit Classifier")

        self.canvas = tk.Canvas(master, width=280, height=280, bg="white")
        self.canvas.pack()

        self.button_predict = tk.Button(master, text="Predict", command=self.predict)
        self.button_predict.pack()

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


# Run the app
root = tk.Tk()
app = DigitClassifierApp(root)
root.mainloop()
