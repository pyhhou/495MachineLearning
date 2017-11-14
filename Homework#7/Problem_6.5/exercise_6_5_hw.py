
# coding: utf-8

# In[ ]:

# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt


# A simple data loading function.

# In[ ]:

### load data
def load_data(csvname):
    data = np.array(np.genfromtxt(csvname, delimiter=','))
    X = data[:,0:-1]
    y = data[:,-1]
    y = np.reshape(y,(np.size(y),1))
    return X,y

# All of the functionality we need to create a gradient descent loop - including functions for computing cost function value, and the descent loop itself.

# In[ ]:

# YOUR CODE GOES HERE -- gradient descent for single layer tanh nn

# sigmoid for softmax/logistic regression minimization
def sigmoid(z):
    y = 1 / (1 + np.exp(-z))
    return y

def compute_cost(c,V,t):
    F = np.tanh(c + np.dot(V, t))
    return F

def gradient_descent(X,y,M):

    # initialize weights and other items
    P = np.size(X, 1)
    b = 1
    w = np.random.randn(M, 1)
    c = np.zeros((M, 1))
    v = np.random.randn(M, 2)
    # M = 4
    alpha = 1e-2
    l_P = np.ones((P,1))

    # stoppers and containers
    max_its = 15000
    k = 1

    x = np.hsplit(X, P)
    ### main ###
    for k in range(max_its):
        # update gradients
        q = np.zeros((P, 1))

        tn = np.zeros((M, P))
        sn = np.zeros((M, P))

        o = np.ones((M, 1))
        for p in np.arange(0, P):
            q[p] = sigmoid(-y[p] * (b + np.dot(o.T, w * np.tanh(c + np.dot(v, x[p])))))

        for m in np.arange(0, M):
            for p in np.arange(0, P):
                tn[m][p] = np.tanh(c[m][0] + np.dot(v[m, :].reshape(1, 2), x[p]))
                sn[m][p] = 1 / np.cosh(c[m][0] + np.dot(v[m, :].reshape(1, 2), x[p])) ** 2

        # YOUR CODE HERE, for grad_b
        grad_b = np.dot(l_P.T, q * (-y))

        grad_w = np.zeros((M, 1))
        grad_c = np.zeros((M, 1))
        grad_v = np.zeros((M, 2))

        tn = np.vsplit(tn, M)
        sn = np.vsplit(sn, M)
        for m in np.arange(0, M):
            grad_w[m] = np.dot(l_P.T, q * (tn[m]).T * -y)
            grad_c[m] = np.dot(l_P.T, q * (sn[m]).T * y) * (-w[m][0])
            grad_v[m, :] = (np.dot(X, q * (sn[m]).T * y) * (-w[m][0])).reshape((2,))

        # take gradient steps
        b = b - alpha * grad_b
        w = w - alpha * grad_w
        c = c - alpha * grad_c
        v = v - alpha * grad_v

        # update stopper and container
        k = k + 1

    return b, w, c, v

# Next our plotting functionality - both plotting points and nonlinear separator.

# In[ ]:

# plot points
def plot_points(X,y):
    ind = np.nonzero(y==1)[0]
    plt.plot(X[ind,0],X[ind,1],'ro')
    ind = np.nonzero(y==-1)[0]
    plt.plot(X[ind,0],X[ind,1],'bo')
    plt.hold(True)

# plot the seprator + surface
def plot_separator(b,w,c,V,X,y):
    s = np.arange(-1,1,.01)
    s1, s2 = np.meshgrid(s,s)

    s1 = np.reshape(s1,(np.size(s1),1))
    s2 = np.reshape(s2,(np.size(s2),1))
    g = np.zeros((np.size(s1),1))

    t = np.zeros((2,1))
    for i in np.arange(0,np.size(s1)):
        t[0] = s1[i]
        t[1] = s2[i]
        F = compute_cost(c,V,t)
        g[i] = np.tanh(b + np.dot(F.T,w))

    s1 = np.reshape(s1,(np.size(s),np.size(s)))
    s2 = np.reshape(s2,(np.size(s),np.size(s)))
    g = np.reshape(g,(np.size(s),np.size(s)))

    # plot contour in original space
    plt.contour(s1,s2,g,1,color = 'k')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.hold(True)


# With everything defined lets run all.

# In[ ]:

# load data
X, y = load_data('genreg_data.csv')
M = 10                  # number of basis functions to use / hidden units

# perform gradient descent to fit tanh basis sum
b,w,c,V = gradient_descent(X.T,y,M)

# plot resulting fit
fig = plt.figure(facecolor = 'white',figsize = (4,4))
plot_points(X,y)
plot_separator(b,w,c,V,X,y)
plt.show()
