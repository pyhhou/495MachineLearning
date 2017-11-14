import numpy as np

def checkSize(w, X, y):
	# w: 785 by 10 matrix
	# X: N by 785 matrix
	# y: N by 1 matrix
	assert y.dtype == 'int'
	assert X.shape[0] == y.shape[0]
	assert X.shape[1] == w.shape[0]
	assert len(y.shape) == 2
	assert y.shape[1] == 1

def loss(w, X, y):
	"""
	Optional
	Useful to run gradient checking
	Utilize softmax function below
	"""
	checkSize(w, X, y)

	# TODO

def grad(w, X, y):
	"""
	Return gradient of multiclass softmax
	Utilize softmax function below
	"""
	checkSize(w, X, y)
	
	# TODO


def softmax(w, X):
	scores = np.matmul(X, w)
	maxscores = scores.max(axis = 1)
	scores = scores - maxscores[:, np.newaxis]
	exp_scores = np.exp(scores)

	sum_scores = np.sum(exp_scores, axis = 1)
	return exp_scores/sum_scores[:, np.newaxis]

def predict(w, X):
	"""
	Prediction
	"""

	# TODO

def accuracy(w, X, y):
	"""
	Accuracy of the model
	"""
	
	# TODO
