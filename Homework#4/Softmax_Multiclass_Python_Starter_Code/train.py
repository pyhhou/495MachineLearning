import numpy as np
from multiClassSoftmax import *

def compactNotation(X):
	"""
	append 1 to X
	"""
	return np.hstack([np.ones([X.shape[0], 1]), X])

def readData(path):
	"""
	Read data from a specified path
	Returns:
		X: in compact notation
		Y: a matrix of [Nsamples, 1] where values are from 0 to 9
	"""

	# TODO

def checkGradient(loss, grad, w, *args):
	"""
	Gradient checking
	"""
	computed_grad = grad(w, *args)

	# compute numerical gradient
	num_grad = np.zeros_like(computed_grad)
	eps = 1e-5
	for i in range(computed_grad.shape[0]):
		for j in range(computed_grad.shape[1]):
			w1 = w.copy()
			w1[i][j] += eps
			num_grad[i][j] = (loss(w1, *args) - loss(w, *args))/eps

	assert np.linalg.norm(computed_grad - num_grad)/np.linalg.norm(computed_grad + num_grad) < 1e-2

def gradientDescent(grad, w0, *args, **kwargs):
	"""
	Gradient descent
	"""
	max_iter = 5000
	alpha = 0.001
	eps = 1e-5

	w = w0
	iter = 0
	while True:
		gradient = grad(w, *args)
		w = w - alpha * gradient

		if iter > max_iter or np.linalg.norm(gradient) < eps:
			break

		if iter  % 1000 == 1:
			print("Iter %d " % iter)

		iter += 1

	return w


if __name__ == "__main__":

	trainX, trainY = readData('MNIST_data/MNIST_train_data.csv')
	testX, testY = readData('MNIST_data/MNIST_test_data.csv')

	# # gradient checking
	# w = np.ones((785,10))
	# print("Checking gradient")
	# checkGradient(loss, grad, w, testX, testY)

	# Optimizing with gradient descent
	w0 = np.random.rand(785,10)
	w_optimal = gradientDescent(grad, w0, testX, testY)

	# Accuracy
	print("Accuracy for training set is %f" % accuracy(w_optimal, trainX, trainY))
	print("Accuracy for test set is %f" % accuracy(w_optimal, testX, testY))