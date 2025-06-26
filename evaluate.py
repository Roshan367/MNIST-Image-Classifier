import numpy as np
from utils import *
from training import *
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = (
    keras.datasets.mnist.load_data()
)

X_test = test_images / 255.0
y_test = test_labels

y_test = to_categorical(y_test)


conv = Convolution((28, 28), 6, 1)
pool = MaxPool(2)
full = Connected(121, 10)

load_model(conv, full)

predictions = []

for data in X_test:
    pred = predict(data, conv, pool, full)
    one_hot_pred = np.zeros_like(pred)
    one_hot_pred[np.argmax(pred)] = 1
    predictions.append(one_hot_pred.flatten())

predictions = np.array(predictions)

print(f"{accuracy_score(predictions, y_test) * 100.0}%")
