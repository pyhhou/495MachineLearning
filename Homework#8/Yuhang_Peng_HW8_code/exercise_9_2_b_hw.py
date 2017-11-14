# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# YOUR CODE GOES HERE -- your PCA function
def your_PCA(X, K):

    C = np.random.rand(X.shape[0],K)
    W = np.random.rand(K,X.shape[1])

    C_new = np.zeros((X.shape[0],K))
    W_new = np.zeros((K,X.shape[1]))

    alpha = 10 ** (-5)
    while np.linalg.norm(C - C_new) > alpha or np.linalg.norm(W - W_new) > alpha:
        C_new = C
        W_new = W
        C = np.dot(np.dot(X,W.T), np.linalg.pinv(np.dot(W,W.T)))
        W = np.dot(np.dot(np.linalg.pinv(np.dot(C.T,C)),C.T),X)

    return C, W

# plot everything
def plot_results(X, C):

    # Print points and pcs
    fig = plt.figure(facecolor = 'white',figsize = (10,4))
    ax1 = fig.add_subplot(121)
    for j in np.arange(0,n):
        plt.scatter(X[0][:],X[1][:],color = 'lime',edgecolor = 'k')

    s = np.arange(C[0,0],-C[0,0],.001)
    m = C[1,0]/C[0,0]
    ax1.plot(s, m*s, color = 'k', linewidth = 2)
    ax1.set_xlim(-.5, .5)
    ax1.set_ylim(-.5, .5)
    ax1.axis('off')

    # Plot projected data
    ax2 = fig.add_subplot(122)
    X_proj = np.dot(C, np.linalg.solve(np.dot(C.T,C),np.dot(C.T,X)))
    for j in np.arange(0,n):
        plt.scatter(X_proj[0][:],X_proj[1][:],color = 'lime',edgecolor = 'k')

    ax2.set_xlim(-.5, .5)
    ax2.set_ylim(-.5, .5)
    ax2.axis('off')

    return

# load in data
X = np.matrix(np.genfromtxt('PCA_demo_data.csv', delimiter=','))
n = np.shape(X)[0]
means = np.matlib.repmat(np.mean(X,0), n, 1)
X = X - means  # center the data
X = X.T
K = 1

# run PCA
C, W = your_PCA(X, K)
# C = np.random.rand(X.shape[0],K)
# plot results
plot_results(X, C)
plt.show()
