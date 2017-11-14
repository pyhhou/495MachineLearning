import numpy as np
import matplotlib.pyplot as plt
import csv

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def read_file(filename):
    reader = csv.reader(open(filename, "rb"), delimiter=",")
    d = list(reader)

    data = np.array(d).astype("float")
    data = np.random.permutation(data)
    X = data[:,0:-1]
    y = data[:,-1]

    y.shape = (len(y),1)

    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    return X,y

def softmax_grad(w, X, y):
    r = -sigmoid(-y*np.dot(X,w))*y
    grad = np.dot(X.T,r)
    return grad

def gradient_descent(w0, X, y):
    max_iter = 100
    alpha = 4.0/(np.linalg.norm(X, ord=2) ** 2)

    w = w0
    iter = 1
    c_gradient = []
    while iter <= max_iter:
        gradient = softmax_grad(w,X,y)
        w = w - alpha * gradient

        c_gradient.append(float(np.dot(np.ones((1,y.shape[0])),np.log(1 + np.exp(-y * np.dot(X,w))))[0][0]))
        iter += 1
    return c_gradient

def stochastic(w0, X, y):
    max_iter = 100
    w = w0
    c_stochastic = []
    iter = 1
    while iter <= max_iter:
        alpha = 1.0 / iter
        for j in range(X.shape[0]):
            gradient = softmax_grad(w,X[j].reshape((1,X.shape[1])),y[j])
            w = w - alpha * gradient
        c_stochastic.append(float(np.dot(np.ones((1,y.shape[0])),np.log(1 + np.exp(-y * np.dot(X,w))))[0][0]))
        iter += 1

    return c_stochastic

X, y = read_file('feat_face_data.csv')
w0 = np.random.randn(497, 1)
w1 = np.random.randn(497, 1)
x = np.arange(100)
c_gradient = []
c_gradient = gradient_descent(w0,X,y)
a1, = plt.plot(x,c_gradient)
c_stochastic = []
c_stochastic = stochastic(w1,X,y)
a2, = plt.plot(x,c_stochastic)
plt.legend(handles=[a1,a2], labels=['gradient descent','stochastic'], loc='upper right')
plt.show()
