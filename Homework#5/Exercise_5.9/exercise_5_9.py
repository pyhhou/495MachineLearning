# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


# a simple data loading function
def load_data():
    data = np.array(np.genfromtxt('noisy_sin_samples.csv', delimiter=','))
    x = np.reshape(data[:,0],(np.size(data[:,0]),1))
    y = np.reshape(data[:,1],(np.size(data[:,1]),1))
    return x,y

# gradient descent for single layer tanh nn 
def tanh_grad_descent(x,y,i):
    # initialize weights and other items
    b, w, c, v = initialize(i)
    P = np.size(x)
    M = 4
    alpha = 1e-3
    l_P = np.ones((P,1))

    # stoppers and containers
    max_its = 15000
    k = 1
    cost_val = []       # container for objective value at each iteration

    ### main ###
    for k in range(max_its):
        # update gradients
        q = np.zeros((P,1))
        for p in np.arange(0,P):
            # YOUR CODE HERE, for vector q
            q[p] = b + np.dot(w.T, np.tanh(c + x[p] * v)) - y[p]

        # YOUR CODE HERE, for grad_b            
        grad_b = 2 * np.dot(l_P.T, q)
        
        grad_w = np.zeros((M,1))
        grad_c = np.zeros((M,1))
        grad_v = np.zeros((M,1))
        for m in np.arange(0,M):
            # YOUR CODE HERE
            t = np.tanh(c[m] + x * v[m])
            s = (1 / np.cosh(c[m] + x * v[m])) ** 2
            grad_w[m] = 2 * np.dot(l_P.T, q * t)
            grad_c[m] = 2 * np.dot(l_P.T, q * s) * w[m]
            grad_v[m] = 2 * np.dot(l_P.T, q * x * s) * w[m]

        # take gradient steps
        b = b - alpha*grad_b
        w = w - alpha*grad_w
        c = c - alpha*grad_c
        v = v - alpha*grad_v

        # update stopper and container
        k = k + 1
        cost_val.append(compute_cost(x,y,b,w,c,v))

    return b, w, c, v, cost_val

# compute cost function value
def compute_cost(x,y,b,w,c,v):
    s = 0
    P = np.size(x)
    for p  in np.arange(0,P):
        s = s + ((b + np.dot(w.T,np.tanh(c + v*x[p])) - y[p])**2)
    return s[0][0]

# initialize parameters - a set of specific initializations for this problem
def initialize(i):
    b = 0
    w = 0
    c = 0
    v = 0
    if (i == 0):
        b = -0.454
        w = np.array([[-0.3461],[-0.8727],[0.6312 ],[0.9760]])
        c = np.array([[-0.6584],[0.7832],[-1.0260],[0.5559]])
        v = np.array([[-0.8571],[-0.8623],[1.0418],[-0.4081]])

    elif (i == 1):
        b = -1.1724
        w = np.array([[.09],[-1.99],[-3.68],[-.64]])
        c = np.array([[-3.4814],[-0.3177],[-4.7905],[-1.5374]])
        v = np.array([[-0.7055],[-0.6778],[0.1639],[-2.4117]])

    else:
        b = 0.1409
        w = np.array([[0.5207],[-2.1275],[10.7415],[3.5584]])
        c = np.array([[2.7754],[0.0417],[-5.5907],[-2.5756]])
        v = np.array([[-1.8030],[0.7578],[-2.4235],[0.6615]])

    return b, w, c, v

# plot tanh approximation 
def plot_approx(b,w,c,v,color):
    M = np.size(c)
    s = np.arange(0,1,.01)
    t = b
    for m in np.arange(0,M):
        t = t + w[m]*np.tanh(c[m] + v[m]*s)

    s = np.reshape(s,np.shape(t))
    plt.plot(s[0],t[0], color = color, linewidth=2)
    plt.xlim(0,1)
    plt.ylim(-1.4,1.4)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y',rotation = 0, fontsize=16)

# plot cost function value at each iteration of gradient descent 
def plot_cost(o, color):
    if np.size(o) == 15000:
        plt.plot(np.arange(100,np.size(o)), o[100:], color = color, linewidth=2)
    else:
        plt.plot(np.arange(1,np.size(o)+1), o, color = color)
    plt.xlabel('iteration', fontsize=14)
    plt.ylabel('cost value', fontsize=14)
    plt.xticks([5000,10000,15000])

# plot data 
def plot_data(x,y):
    plt.scatter(x,y,s=30,color='k')
    
# load data
x, y = load_data()

# plot data
fig = plt.figure(figsize = (8,4))
plt.subplot(1,2,1)
plot_data(x,y)

# perform gradient descent to fit tanh basis sum
num_runs = 3
colors = ['r','g','b']
for i in np.arange(0,num_runs):

    # minimize least squares cost
    b,w,c,v,cost_val = tanh_grad_descent(x,y,i)

    # plot resulting fit to data
    color = colors[i]
    plt.subplot(1,2,1)
    plot_approx(b,w,c,v,color)

    # plot objective value decrease for current run
    plt.subplot(1,2,2)
    plot_cost(cost_val,color)

plt.show()