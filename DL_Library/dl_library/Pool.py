#===============================
#	Neural Network (version 4)
#
#	Pool :
#		* Pooling layer
#
#===============================

# Import external libraries
import numpy as np

class Pool(object):

	def __init__(self, hyper_parameters):

		# Hyperparameters
		self.f = hyper_parameters["f"]
		self.stride = hyper_parameters["stride"]
		self.mode = hyper_parameters["mode"]

		# Cache
		self.A_prev = None

	def forward(self, A_prev):
		(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

		f = self.f
		stride = self.stride

		n_H = int((n_H_prev - f) / stride) + 1
		n_W = int((n_W_prev - f) / stride) + 1
		n_C = n_C_prev

		A = np.zeros((m, n_H, n_W, n_C))

		for i in range(m):
			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						vert_start = h * stride
						vert_end = vert_start + f
						horiz_start = w * stride
						horiz_end = horiz_start + f

						a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

						if self.mode == "max":
							A[i, h, w, c] = np.max(a_prev_slice)
						elif self.mode == "average":
							A[i, h, w, c] = np.mean(a_prev_slice)

		self.A_prev = A_prev

		return A

	def backward(self, dA):
		f = self.f
		stride = self.stride

		m, n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
		m, n_H, n_W, n_C = dA.shape

		dA_prev = np.zeros(self.A_prev.shape)

		for i in range(m):
			a_prev = self.A_prev[i]
			for h in range(n_H):
				for w in range(n_W):
					for c in range(n_C):
						# Maybe delete the stride because we are going backward, so we're not passing values
						vert_start = h * stride
						vert_end = vert_start + f
						horiz_start = w * stride
						horiz_end = horiz_start + f

						if self.mode == "max":
							a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
							mask = self.create_mask_from_window(a_prev_slice)
							dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])

						if self.mode == "average":
							da = dA[i, h, w, c]
							shape = (f, f)
							dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += self.distribute_value(da, shape)

		return dA_prev

	def create_mask_from_window(self, x):
		mask = x == np.max(x)

		return mask

	def distribute_value(self, dZ, shape):
		(n_H, n_W) = shape

		average = dZ / (n_H * n_W)
		A = np.ones(shape) * average

		return A