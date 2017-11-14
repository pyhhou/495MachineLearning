import numpy as np
import matplotlib.pyplot as plt
import csv

def read_data(filename):
	'''
	Read data from csvfile

	Parameters:
	----------------------------------------------------
	filename: path to data file

	Returns:
	----------------------------------------------------
	X: ndarray of shape (1, P)
	y: ndarray of shape (P, 1)
	
	Hints:
	----------------------------------------------------
	Make sure the shapes of X, y conform to above. You may
	find "np.genfromtxt(, delimiter=','), np.newaxis" useful

	'''
	####################################################
	###                  Your Code                   ###
	####################################################
	reader = csv.reader(open(filename, "r"), delimiter = ",")
	d = list(reader)
	data = np.array(d).astype("float")
	X = data[:,0];
	y = data[:,1];
	X.shape = (1, len(X))
	y.shape = (len(y), 1)
	####################################################
	###                    End                       ###
	####################################################
	assert X.shape == (1, data[:, 0].size), "Shape of X incorrect"
	assert y.shape == (data[:,-1].size, 1), "Shape of y incorrect"
	return X, y

def fourier_basis(X, D):
	'''
	Return Fourier basis for X (with ONE bias dimension)

	Parameters:
	----------------------------------------------------
	X: data ndarray of shape (1, P)
	D: degree of Fourier basis features

	Returns:
	----------------------------------------------------
	F: ndarray of shape (2D+1, P)
	
	Hints:
	----------------------------------------------------
	Make sure the shapes of F conform to above. You may
	find "np.arange, np.reshape, np.concatenate(, axis=0),
	np.ones, np.cos, np.sin"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	f = np.ones((1, X.size))
	m = np.arange(1, D + 1).reshape(D, 1)
	w = 2 * np.pi * np.dot(m, X) 
	F_1 = np.cos(w)
	F_2 = np.sin(w)

	F = np.concatenate((f, F_1, F_2), axis = 0)
	####################################################
	###                    End                       ###
	####################################################
	assert F.shape == (2*D+1, X.size), "Shape of F incorrect"
	return F

def poly_basis(X, D):
	'''
	Return polynomial basis for X (with ONE bias dimension)

	Parameters:
	----------------------------------------------------
	X: data ndarray of shape (1, P)
	D: degree of Fourier basis features

	Returns:
	----------------------------------------------------
	F: ndarray of shape (D+1, P)
	
	Hints:
	----------------------------------------------------
	Make sure the shapes of F conform to above. You may
	find "np.arange, np.reshape, np.power"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	f = np.ones((1, X.size))
	m = np.arange(1, D + 1).reshape(D, 1)
	y = np.power(X, m)
	# for d in range(1,D+1):
	# 	F = np.concatenate((F,np.power(X,d)), axis = 0)
	# F.shape = (D + 1, X.size)
	F = np.concatenate((f, y), axis = 0)
	####################################################
	###                    End                       ###
	####################################################
	assert F.shape == (D+1, X.size), "Shape of F incorrect"
	return F

def least_square_sol(F, y):
	'''
	Refer to eq. 5.19 in the text

	Parameters:
	----------------------------------------------------
	F: ndarray of shape (2D+1 or D+1 depends on what basis, P)
	y: ndarray of shape (P, 1)

	Returns:
	----------------------------------------------------
	w: learned weighter vector of shape (2D+1, 1)
	
	Hints:
	----------------------------------------------------
	You may find "np.linalg.pinv, np.dot"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	w = np.dot(np.linalg.pinv(np.dot(F, F.T)), np.dot(F,y))

	####################################################
	###                    End                       ###
	####################################################
	assert w.shape == (F.shape[0], 1), "Shape of w incorrect"
	return w

def mean_square_error(w, F, y):
	'''
	Refer to eq. 5.19 in the text

	Parameters:
	----------------------------------------------------
	w: learned weighter vector of shape (2D+1, 1)
	F: ndarray of shape (2D+1, P)
	y: ndarray of shape (P, 1)

	Returns:
	----------------------------------------------------
	mse: a scaler, mean square error of your learned model
	
	Hints:
	----------------------------------------------------
	You may find "np.dot, np.mean"  useful

	'''	
	####################################################
	###                  Your Code                   ###
	####################################################
	mse = np.mean(np.dot((np.dot(F.T, w) - y).T, (np.dot(F.T, w) - y)))

	# temp = np.power(np.dot(F.T,w) - y, 2)
	# mse = np.mean(temp)

	####################################################
	###                    End                       ###
	####################################################
	return mse

def random_split(P, K):
	'''
	Return a list of K arrays, each of which are indices 
	of data point
	
	Parameters:
	----------------------------------------------------
	P: number of data points
	K: number of folds

	Returns:
	----------------------------------------------------
	folds: a list of K arrays, each of which are position 
	indices of data point
	
	Hints:
	----------------------------------------------------
	You may find "np.split, np.random.permutation, np.arange" 
	useful	

	'''
	####################################################
	###                  Your Code                   ###
	####################################################
	folds = np.arange(P)
	folds = np.random.permutation(folds)
	folds = np.split(folds, K)

	####################################################
	###                    End                       ###
	####################################################
	assert len(folds) == K, 'Number of folds incorrect'
	return folds

def train_val_split(X, y, folds, fold_id):
	'''
	Split the data into training and validation sets

	Parameters:
	----------------------------------------------------
	X: ndarray of shape (1, P)
	y: ndarray of shape (P, 1)
	folds: a list of K arrays, each of which are indices 
	of data point
	fold_id: the id of the fold you want to be validation set

	Returns:
	----------------------------------------------------
	X_train: training set of X
	y_train: training label
	X_val: validation set of X
	y_val: validation label

	'''
	####################################################
	###                  Your Code                   ###
	####################################################
	X_val = X[0][folds[fold_id]]
	X_val.shape = (1, X_val.size)
	y_val = y[folds[fold_id]]
	y_val.shape = (y_val.size, 1)

	X_train = np.delete(X, folds[fold_id])
	X_train.shape = (1, X_train.size)
	y_train = np.delete(y, folds[fold_id])
	y_train.shape = (y_train.size, 1)

	####################################################
	###                    End                       ###
	####################################################
	assert y_val.size + y_train.size == y.size, 'Split incorrect'
	return X_train, y_train, X_val, y_val

def make_plot(D, MSE_train, MSE_val):
	plt.figure()
	train, = plt.plot(D, MSE_train, 'yv--')
	val, = plt.plot(D, MSE_val, 'bv--')
	plt.legend(handles=[train, val], labels=['training_error', 'validation error'], loc='upper left')
	plt.xlabel('Degree of Fourier basis')
	plt.ylabel('Error in log scale')
	plt.yscale('log')
	plt.show()
