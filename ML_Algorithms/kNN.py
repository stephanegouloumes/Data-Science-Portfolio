import numpy as np

class kNN(object):

	def fit(self, X, Y):
		self.X = X
		self.Y = Y

	def compute_distances(self, input_x, type):
		distances = np.empty([self.X.shape[0], 2])

		num_features = input_x.shape[0]
		if type == "euclidean":
			for i, x in enumerate(self.X):
				distances[i, 0] = i
				distances[i, 1] = np.sqrt(np.sum(np.square(x - input_x)))

		return distances

	def predict(self, X, k = 1, type = "euclidean"):

		# Compute the distances between the input and the data
		distances = self.compute_distances(X, type)

		# Sort by distance
		distances = distances[distances[:, 1].argsort()]

		# Get the kNN
		knn = {}
		for i in range(k):
			if self.Y[int(distances[i, 0])] in knn:
				knn[self.Y[int(distances[i, 0])]] += 1
			else:
				knn[self.Y[int(distances[i, 0])]] = 1
		
		max_value = max(knn.values())
		prediction = [k for k, v in knn.items() if v == max_value]
		return prediction[0]

### TEST 1 : Classification

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(123)

digits = load_digits()
X, Y = digits.data, digits.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Example digits
# fig = plt.figure(figsize=(10,8))
# for i in range(10):
#     ax = fig.add_subplot(2, 5, i+1)
#     plt.imshow(X[i].reshape((8,8)), cmap='gray')

# plt.show()

knn = kNN()
knn.fit(X_train, Y_train)

predictions = []
for i in range(200):
	predictions.append(knn.predict(X_test[i], k = 5))

print(np.mean(predictions == Y_test[:200]))