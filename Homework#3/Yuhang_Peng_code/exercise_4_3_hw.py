# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

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
    X = data[:,0:2]
    y = data[:,2]
    y.shape = (len(y),1)
    
    # pad data with ones for more compact gradient computation
    o = np.ones((np.shape(X)[0],1))
    X = np.concatenate((o,X),axis = 1)
    X = X.T
    
    return X,y

# YOUR CODE GOES HERE - create a gradient descent function for softmax cost/logistic regression 
def softmax_grad(X,y):
    alpha = 2
    w0 = array([2,X[1][10],X[2][10]])
    w0.shape = (3,1)
    w = w0
    # x_p = np.hsplit(X,len(X[0]))
    # start gradient descent loop
    grad = 1
    iter = 1
    max_its = 500
    while iter <= max_its:
        # take gradient step
        r = -y * sigmoid(-y * dot(X.T,w))
        grad = np.dot(X,r)
        w = w - alpha*grad
        iter += 1

    # print w
    return w

# plots everything 
def plot_all(X,y,w):
    # custom colors for plotting points
    red = [1,0,0.4]  
    blue = [0,0.4,1]
    
    # scatter plot points
    fig = plt.figure(figsize = (4,4))
    ind = np.argwhere(y==1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1,ind],X[2,ind],color = red,edgecolor = 'k',s = 25)
    ind = np.argwhere(y==-1)
    ind = [s[0] for s in ind]
    plt.scatter(X[1,ind],X[2,ind],color = blue,edgecolor = 'k',s = 25)
    plt.grid('off')
    
    # plot separator
    s = np.linspace(0,1,100) 
    plt.plot(s,(-w[0]-w[1]*s)/w[2],color = 'k',linewidth = 2)
    
    # clean up plot
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.show()
    
# load in data
X,y = load_data('imbalanced_2class.csv')
# print X

# run gradient descent
w = softmax_grad(X,y)

# plot points and separator
plot_all(X,y,w)