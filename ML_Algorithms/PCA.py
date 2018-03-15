import numpy as np

class PCA(object):

	def normalize_data(self, X):
		return (X - np.mean(X)) / np.std(X)

	def fit(self, X):
		# Compute the covariance matrix
		matrix = np.matrix(X)
		cov = (matrix.T * matrix) / matrix.shape[0]

		# SVD (Singular-Value Decomposition)
		U, S, V = np.linalg.svd(cov)

		return U, S, V

	def project_data(self, X, U, k):
		U_reduced = U[:, :k]
		return np.dot(X, U_reduced)

	def recover_data(self, Z, U, k):
		U_reduced = U[:, :k]
		return np.dot(Z, U_reduced.T)


### TEST 1 : 2D data

import matplotlib.pyplot as plt

# Create training data
X1 = np.linspace(0, 5, 100) 
X2 = 1.5 * X1 + 2
noise = np.random.normal(0, 1, 100) 
X2 = X2 + noise

X1 = X1.reshape(X1.shape[0], 1)
X2 = X2.reshape(X2.shape[0], 1)

X = np.concatenate((X1, X2), axis = 1)

pca = PCA()
X_normalized = pca.normalize_data(X)
U, S, V = pca.fit(X_normalized)

# Plot the eigenvectors
means = np.mean(X_normalized, axis = 0)
stds = np.std(X_normalized, axis = 0)

plot = plt.scatter(X_normalized[:,0], X_normalized[:,1], s = 30, facecolors = 'none', edgecolors = 'b')
plt.title("Eigenvectors")
plt.xlabel('X1')
plt.ylabel('X2')

plt.plot([means[0], means[0] + 1.5 * S[0] * U[0,0]], [means[1], means[1] + 1.5 * S[0] * U[0,1]], color = 'red', linewidth = 2, label = 'First Principal Component')
plt.plot([means[0], means[0] + 1.5 * S[1] * U[1,0]], [means[1], means[1] + 1.5 * S[1] * U[1,1]], color = 'green', linewidth = 2, label = 'Second Principal Component')
leg = plt.legend(loc = 4)

plt.show()

# Project the data on 1 dimension
Z = pca.project_data(X_normalized, U, 1)

# Recover the projected data
X_recovered = pca.recover_data(Z, U, 1)

# Plot the data before and after PCA
fig = plt.figure(figsize = (12, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.scatter(X_normalized[:, 0], X_normalized[:, 1])
ax1.set_title("Before PCA")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")

ax2.scatter([X_recovered[:, 0]], [X_recovered[:, 1]])
ax2.set_title("After PCA")
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")

plt.show()