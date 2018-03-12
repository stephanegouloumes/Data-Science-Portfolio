#===============================
#	Neural Network (version 4)
#
#	Main
#
#===============================

# Import the library
import dl_library as dl

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
import keras

### TESTING 1 (Simple use of a deep layer network)

X_train = np.array([[1, 2], [3, 4], [2, 1], [4, 3], [5, 6], [0, 1]])
Y_train = np.array([[1, 0, 1, 0, 0, 1]])
X_train = X_train.T

X_test = np.array([[0, 1], [1, 1], [5, 4], [3, 3], [2, 2], [2, 1]])
Y_test = np.array([[1, 1, 0, 0, 1, 1]])
X_test = X_test.T

model = dl.Model()
model.add_layer(dl.Dense([2, 20], dl.Activation("relu")))
model.add_layer(dl.Dense([20, 1], dl.Activation("sigmoid")))

model.fit(X_train, Y_train, learning_rate = 0.5, num_iterations = 5000, print_cost = True)
model.predict(X_train, Y_train, X_test, Y_test)

### TESTING 2 (Simple classification with Softmax)

X, Y = make_classification(n_samples = 1000, n_features = 2, n_redundant = 0, n_informative = 2, random_state = 42)

# Scatter plot of the data set
plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")
# plt.show()

Y = keras.utils.to_categorical(Y, num_classes = 2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

model = dl.Model()
model.add_layer(dl.Dense([2, 20], dl.Activation("relu")))
model.add_layer(dl.Dense([20, 15], dl.Activation("relu")))
model.add_layer(dl.Dense([15, 10], dl.Activation("relu")))
model.add_layer(dl.Dense([10, 2], dl.Activation("softmax")))

model.fit(X_train, Y_train, learning_rate = 0.5, num_iterations = 10000, print_cost = True)
model.predict(X_train, Y_train, X_test, Y_test)

### TESTING 3 (MNIST on Dense + Softmax)

train = pd.read_csv('../../Kaggles/data/mnist/train.csv')
test = pd.read_csv('../../Kaggles/data/mnist/test.csv')

X_train = train.drop(labels = ['label'], axis = 1)
Y_train = train['label']

X_train = X_train[:100]
Y_train = Y_train[:100]

X_train = X_train / 255.0
test = test / 255.0

Y_train = keras.utils.to_categorical(Y_train, num_classes = 10)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 42)

X_train = X_train.T
X_test = X_test.T
Y_train = Y_train.T
Y_test = Y_test.T

model = dl.Model()
model.add_layer(dl.Dense([784, 50], dl.Activation("relu")))
model.add_layer(dl.Dense([50, 10], dl.Activation("softmax")))

model.fit(X_train, Y_train, learning_rate = 0.5, num_iterations = 500, print_cost = True)
model.predict(X_train, Y_train, X_test, Y_test)

### TESTING 4

train = pd.read_csv('../Kaggles/data/mnist/train.csv')
test = pd.read_csv('../Kaggles/data/mnist/test.csv')

X_train = train.drop(labels = ['label'], axis = 1)
Y_train = train['label']

X_train = X_train[:100]
Y_train = Y_train[:100]

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

Y_train = keras.utils.to_categorical(Y_train, num_classes = 10)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 42)

Y_train = Y_train.T
Y_test = Y_test.T

output_conv = int((28 - 3 + 2 * 2) / 2) + 1
output_conv = int((output_conv - 3 + 2 * 0) / 2) + 1
output_pool = int((output_conv - 2) / 2) + 1
output_final = output_pool * output_pool * 2

model = dl.Model()
model.add_layer(dl.Conv2D([3, 3, 1, 32], {"pad" : 2, "stride" : 2}, dl.Activation("relu")))
model.add_layer(dl.Pool({"f" : 2, "stride" : 2, "mode" : "max"}))
model.add_layer(dl.Dense([1568, 10], dl.Activation("softmax"), True))

model.fit(X_train, Y_train, learning_rate = 0.5, num_iterations = 500, print_cost = True)