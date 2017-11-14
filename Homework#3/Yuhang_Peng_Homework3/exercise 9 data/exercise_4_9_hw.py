from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import csv

# sigmoid for softmax/logistic regression minimization
def sigmoid(z): 
    y = 1/(1+np.exp(-z))
    return y
    
# import training data 
def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "rb"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")
    X = data[:,0:8]
    y = data[:,8]
    y.shape = (len(y),1)
    
    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    X = X.T
    
    return X,y

# create a gradient descent function for softmax cost/logistic regression 
def softmax(X,y):
    # alpha = 0.2
    w0 = array([0,0,0,0,0,0,0,0,0]) 
    w0.shape = (9,1)
    # w1 = w0
    w1 = w0         # for softmax
    # start gradient descent loop
    # grad = 1
    iter = 1
    max_its = 15
    mismatch_soft_list = []
    # x = np.hsplit(X,len(X[0]))
    while iter <= max_its:
        # take gradient step to softmax
        r = -sigmoid(- y * dot(X.T,w1)) * y
        r_2 = sigmoid(- y * dot(X.T,w1)) * (1 - sigmoid(- y * dot(X.T,w1)))
        grad_soft = np.dot(X,r)
        # grad_soft_2 = dot(dot(X,np.diagflat(r_2)),X.T)
        grad_soft_2 = dot(dot(X,diagflat(r_2)),X.T)
        w1 = w1 - dot(np.linalg.pinv(grad_soft_2),grad_soft)
        soft_mis = count_mis(X,y,w1)
        mismatch_soft_list.append(soft_mis)

        iter += 1
    return mismatch_soft_list

def square_margin(X,y):
    # alpha = 0.2
    w0 = array([0,0,0,0,0,0,0,0,0]) 
    w0.shape = (9,1)
    # w1 = w0
    w2 = w0         # for square
    # start gradient descent loop
    # grad = 1
    iter = 1
    max_its = 15
    mismatch_square_list = []
    # x = np.hsplit(X,len(X[0]))
    while iter <= max_its:
        # print grad_soft
        # print grad_soft_2
        # take gradient step to square
        grad_square = -2 * dot((X * y.T), np.maximum(0, 1 - y * dot(X.T,w2)))
        temp = 1 - y * dot(X.T,w2)
        temp = np.reshape(temp, len(y))         # change into one dimension array
        idx = np.where(temp <= 0)               # find the index of element = 0
        _X = X.T
        _X[idx] = 0                             # do not consider that -y(b+xw) < 0
        grad_square_2 = 2 * dot(_X.T, _X)
        w2 = w2 - dot(np.linalg.pinv(grad_square_2),grad_square)
        square_mis = count_mis(X,y,w2)
        mismatch_square_list.append(square_mis)

        iter += 1
    return mismatch_square_list

def count_mis(X,y,w):    
    count = 0
    for i in range(699):    # **
        if y[i] * dot(X.T,w)[i][0] < 0:
            count += 1
    return count

# plots everything 
def plot_all(soft,square):
    # custom colors for plotting points
    x_axis = [i for i in range(1,16)]
    plt.plot(x_axis,soft,'r',label = "softmax")
    plt.plot(x_axis,square,'g',label = "squared_margin")
    plt.xlabel('Number of iterations')
    plt.ylabel('Misclassification')
    legend()
    plt.show()
    
# load in data
X,y = load_data('breast_cancer_data.csv')
# print X

# run gradient descent
mismatch_soft_list = softmax(X,y)
mismatch_square_list = square_margin(X,y)
print mismatch_soft_list
# print mismatch_square_list
# plot points and separator
plot_all(mismatch_soft_list,mismatch_square_list)