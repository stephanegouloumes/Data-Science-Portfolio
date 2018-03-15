import numpy as np
import random

'''
	Kmeans :
		* Randomly initialize n centroids
		* For each x in X
			* Compute distances between each centroid and x
			* Assign x to the nearest centroid
			* Recompute the centroid by taking x into account
'''

class kMeans(object):

	def fit(self, X, Y, n_clusters):

		# The data points
		self.data = X
		self.data_cluster = np.zeros((X.shape[0], 1), dtype = int)
		self.data_cluster -= 1 # Assign the cluster -1 to every data point

		# Y
		self.Y = Y

		# The total number of centroids
		self.n_clusters = n_clusters

		# Initialize each centroid by randomly assigning it a data point
		indices = random.sample(range(self.data.shape[0]), self.n_clusters)
		self.centroids = np.array(self.data[indices])
		for i in range(n_clusters):
			self.data_cluster[indices[i]] = i

		previous_clusters = None

		iteration = 0
		while True:
			# Compute the distances between each data point and the centroids
			new_clusters = []
			for i, x in enumerate(self.data):
				distances = self.compute_distances(x)
				ind = np.unravel_index(np.argmin(distances, axis = None), distances.shape)
				ind = ind[0]
				new_clusters.append(ind)
				self.data_cluster[i] = ind

			if new_clusters == previous_clusters:
				return
			
			previous_clusters = new_clusters

			# Compute the new centroids
			for k in range(self.n_clusters):
				self.centroids[k] = np.mean([self.data[i] for i, y in enumerate(self.data_cluster[:, 0]) if y == k], axis = 0)
			
			iteration += 1
			if iteration % 10 == 0:
				print("Iteration (" + str(iteration + 1) + ")")

	def compute_distances(self, x):
		distances = np.sqrt(np.sum(np.square(self.centroids - x), axis = 1))
		return distances

	def plot_clusters(self):
		plt.scatter(self.data[:, 0], self.data[:, 1], marker = '.', c = self.Y)
		plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c = 'r')
		plt.show()


### TEST 1

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
np.random.seed(123)

X, Y = make_blobs(centers = 4, n_samples = 1000)
# print("Shape of dataset: " + str(X.shape))

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.title("Dataset with 4 clusters")
plt.xlabel("First feature")
plt.ylabel("Second feature")
# plt.show()

km = kMeans()
km.fit(X, Y, 4)
km.plot_clusters()