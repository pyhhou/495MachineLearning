import numpy as np
import matplotlib.pyplot as plt
import csv

def read_data(filename):
	reader = csv.reader(open(filename, "r"), delimiter = ",")
	d = list(reader)
	data = np.array(d).astype("float")
	x = data[:,0]
	y = data[:,1]
	label = data[:,2]
	x.shape = (len(x), 1)
	y.shape = (len(y), 1)
	label.shape = (len(label), 1)

	assert x.shape == (data[:, 0].size, 1), "Shape of X incorrect"
	assert y.shape == (data[:, 1].size, 1), "Shape of y incorrect"
	assert label.shape == (data[:, 2].size, 1), "Shape of label incorrect"

	return x, y, label

def poly_basis(x, y, D):
	# f = np.ones((1, X.size))
	# P = x.size
	M = [2, 5, 9, 14, 20, 27, 35, 44]
	F = []
	for p in range(x.size):
		f_m = []
		for dx in range(0, D + 1):
			for dy in range(0, D + 1):
				if dx + dy > d:
					continue
				f_m.append((x[p] ** dx) * (y[p] ** dy))
		f_m = np.array(f_m)
		f_m.shape = (1, M[D - 1] + 1)
		# if p == 0:
		F.append(f_m)
		# F = np.concatenate((F, f_m), axis = 0)
	F = np.array(F).reshape(x.size, M[D - 1] + 1)
	assert F.shape == (x.size, M[D - 1] + 1), "Shape of F incorrect"
	return F.T

def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return y

def softmax_gradient_decent(F, label, D):
	# initialize weights and other items
	# P = np.size(label, 1)
	# b = 1
	# alpha = 1e-2
	# l_P = np.ones((P,1))
	M = [2, 5, 9, 14, 20, 27, 35, 44]
	w = np.random.randn(M[D - 1] + 1, 1)
 #    # stoppers and containers
	# max_its = 15000
	# k = 1

	# f_h = np.hsplit(F, P)
	# f_v = np.hsplit(F.T, M[D - 1] + 1)
 #    ### main ###
	# for k in range(max_its):
 #        # update gradients
	# 	q = np.zeros((P, 1))
	# 	for p in np.arange(0, P):
	# 		q[p] = sigmoid(-label[p] * (b + np.dot((f_h[p]).T, w)))

 #        # for grad_b
	# 	grad_b = np.dot(l_P.T, q * (-y))

	# 	grad_w = np.zeros((M[D - 1] + 1, 1))
	# 	for m in np.arange(0, M[D - 1] + 1):
	# 		grad_w[m] = np.dot(l_P.T, q * f_v[m] * -y)

 #        # take gradient steps
	# 	b = b - alpha * grad_b
	# 	w = w - alpha * grad_w

 #        # update stopper and container
	# 	k = k + 1

	# assert w.shape == (F.shape[0], 1), "Shape of w incorrect"
	alpha = 1e-3
    # x_p = np.hsplit(X,len(X[0]))
    # start gradient descent loop
	grad = 1
	iter = 1
	max_its = 4000000
	while iter <= max_its:
        # take gradient step
		r = -label * sigmoid(-label * np.dot(F.T, w))
		grad = np.dot(F,r)
		w = w - alpha * grad
		iter += 1
	return w

def softmax_netwon_method(F, label, D):
	# alpha = 0.2
	# w1 = w0
	M = [2, 5, 9, 14, 20, 27, 35, 44]
	w1 = np.random.randn(M[D - 1] + 1, 1) / 100
    # start gradient descent loop
    # grad = 1
	iter = 1
	max_its = 6
	while iter <= max_its:
        # take gradient step to softmax
		r = -sigmoid(- label * np.dot(F.T,w1)) * label
		r_2 = sigmoid(- label * np.dot(F.T,w1)) * (1 - sigmoid(- label * np.dot(F.T,w1)))
		grad_soft = np.dot(F,r)
        # grad_soft_2 = dot(dot(X,np.diagflat(r_2)),X.T)
		# grad_soft_2 = np.dot(np.dot(F,np.diagflat(r_2)),F.T)
		grad_soft_2 = np.dot(F, r_2 * F.T)
		w1 = w1 - np.dot(np.linalg.pinv(grad_soft_2),grad_soft)

		iter += 1
	return w1

def mean_softmax_error(w, F, label):
	mse = np.mean(np.maximum(0, -label * np.dot(F.T, w)))

	# temp = np.power(np.dot(F.T,w) - y, 2)
	# mse = np.mean(temp)

	return mse

def random_split(P, K):
	folds = np.arange(P)
	folds = np.random.permutation(folds)
	folds = np.array_split(folds, K)

	assert len(folds) == K, 'Number of folds incorrect'
	return folds

def train_val_split(x, y, lable, folds, fold_id):
	x_val = x[folds[fold_id]]
	x_val.shape = (x_val.size, 1)
	y_val = y[folds[fold_id]]
	y_val.shape = (y_val.size, 1)
	label_val = label[folds[fold_id]]
	label_val.shape = (label_val.size, 1)

	x_train = np.delete(x, folds[fold_id])
	x_train.shape = (x_train.size, 1)
	y_train = np.delete(y, folds[fold_id])
	y_train.shape = (y_train.size, 1)
	label_train = np.delete(label, folds[fold_id])
	label_train.shape = (label_train.size, 1)

	assert x_val.size + x_train.size == x.size, 'Split incorrect'
	assert y_val.size + y_train.size == y.size, 'Split incorrect'
	assert label_val.size + label_train.size == label.size, 'Split incorrect'

	return x_train, y_train, x_val, y_val, label_val, label_train

def make_plot(D, MSE_train, MSE_val):
	plt.figure()
	train, = plt.plot(D, MSE_train, 'yv--')
	val, = plt.plot(D, MSE_val, 'bv--')
	plt.legend(handles=[train, val], labels=['training_error', 'validation error'], loc='upper left')
	plt.xlabel('Degree of Ploy basis')
	plt.ylabel('Error')
	# plt.yscale('log')
	plt.show()

x, y, label = read_data('2eggs_data.csv')
num_fold, num_degree = 3, 8

# random the index, then split
folds = random_split(P=y.size, K=num_fold)

MSE_train, MSE_val = [0]*num_degree, [0]*num_degree
D = np.arange(1, num_degree + 1)
for f in xrange(num_fold):
	# split data for each fold
	x_train, y_train, x_val, y_val, label_val, label_train = train_val_split(x, y, label, folds, fold_id = f)
	for i, d in enumerate(D):
		# Get the polynomial type f for training and testing data
		F_train = poly_basis(x_train, y_train, D = d)
		F_val = poly_basis(x_val, y_val, D = d)
		# Do the gradient decent to obtain the optimal b and w
		w = softmax_gradient_decent(F_train, label_train, D = d)
		# w = softmax_netwon_method(F_train, label_train, D = d)
		# Calculate the error
		MSE_train[i] += mean_softmax_error(w, F_train, label_train) / num_fold
		MSE_val[i] += mean_softmax_error(w, F_val, label_val) / num_fold

print 'The best degree of polynomial basis, in terms of validation error, is %d' % (MSE_val.index(min(MSE_val))+1)

# plot the result
make_plot(D, MSE_train, MSE_val)
