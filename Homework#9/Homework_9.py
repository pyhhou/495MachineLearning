import numpy as np
import matplotlib.pyplot as plt
import csv

def read_train_data(filename):
	reader = csv.reader(open(filename, "r"), delimiter = ",")
	d = list(reader)
	data = np.array(d).astype("float")
	# print d
	m = len(d)
	n = len(d[0])
	# print m, n
	x = data[:, :-1]
	label = data[:, - 1]
	x.shape = (m, n - 1)
	label.shape = (len(label), 1)
	label = label - 1

	assert x.shape == (data[:, 0].size, n - 1), "Shape of X incorrect"
	assert label.shape == (data[:, n - 1].size, 1), "Shape of label incorrect"

	return x, label

def read_test_data(filename):
	reader = csv.reader(open(filename, "r"), delimiter = ",")
	d = list(reader)
	data = np.array(d).astype("float")
	# print d
	m = len(d)
	n = len(d[0])
	# print m, n
	x = data[:,:]
	# label = data[:, n - 1]
	x.shape = (m, n)
	# label.shape = (len(label), 1)

	assert x.shape == (data[:, 0].size, n), "Shape of X incorrect"
	# assert label.shape == (data[:, n - 1].size, 1), "Shape of label incorrect"

	return x

def sigmoid(t):
	return 1/(1 + np.exp(-t))

def checkSize(w, X, y):
	# w and y are column vector, shape [N, 1] not [N,]
	# X is a matrix where rows are data sample
	assert X.shape[0] == y.shape[0]
	assert X.shape[1] == w.shape[0]
	assert len(y.shape) == 2
	assert len(w.shape) == 2
	assert w.shape[1] == 1
	assert y.shape[1] == 1

def compactNotation(X):
	return np.hstack([np.ones([X.shape[0], 1]), X])

def softmaxGrad(w, X, y):
	checkSize(w, X, y)
	# RETURN GRADIENT
	r = -sigmoid(-y*np.dot(X,w))*y
	grad = np.dot(X.T,r)

	return grad

def output(OVA, X):
	"""
	Calculate accuracy using matrix operations!
	"""
	temp = np.dot(X,OVA)
	res = np.argmax(temp, axis = 1)
	# res.shape=(len(y),1)
	# return np.sum(y==res) / len(y)
	return res + 1

def gradientDescent(grad, w0, *args, **kwargs):
	max_iter = 5000
	alpha = 0.001
	eps = 10**(-5)

	w = w0
	iter = 0
	while True:
		# print(iter)
		gradient = grad(w, *args)
		w = w - alpha * gradient

		if iter > max_iter or np.linalg.norm(gradient) < eps:
			break
		if iter  % 1000 == 1:
			print("Iter %d " % iter)
		iter += 1
	return w

def oneVersusAll(Y, value):
	"""
	generate label Yout,
	where Y == value then Yout would be 1
	otherwise Yout would be -1
	"""
	Yout = []
	for i in range(len(Y)):
		if Y[i][0] == value:
			Yout.append(1)
		else:
			Yout.append(-1)
	Yout = np.array(Yout)
	Yout.shape = (len(Y),1)
	return Yout

if __name__=="__main__":
	trainX, trainY = read_train_data('train_data_label.csv')
	# training individual classifier
	Nfeature = trainX.shape[1]
	Nclass = 4
	OVA = np.zeros((Nfeature, Nclass))
	for i in range(Nclass):
		print("Training for class " + str(i))
		w0 = np.random.rand(Nfeature,1)
		OVA[:, i:i+1] = gradientDescent(softmaxGrad, w0, trainX, oneVersusAll(trainY, i))

	# print OVA
	testX = read_test_data('test_data.csv')
	test_label = output(OVA, testX)
	test_label.shape = (len(test_label), 1)
	print test_label

	resultFile = open("Yuhang_Peng_Homework_9.csv",'wb')
	wr = csv.writer(resultFile)
	wr.writerows(test_label)


# x, label = read_data('train_data_label.csv')

# (m, n) = x.shape
# print m, n
