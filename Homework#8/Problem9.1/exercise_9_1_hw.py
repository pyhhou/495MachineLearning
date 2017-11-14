# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

# YOUR CODE GOES HERE
def your_K_means(X, K, C0):
    C = C0
    P = X.shape[1]
    W = np.zeros((K, P))

    i = 0
    while i < P:
        j = 0
    	min_diff = 100000000
        min_index = -1
        while j < K:
            if np.linalg.norm(X[:,i] - C[:,j]) < min_diff:
                min_diff = np.linalg.norm(X[:,i] - C[:,j])
                min_index = j
            j += 1
        W[min_index][i] = 1
        i += 1
    C = np.zeros((X.shape[0],K))

    i = 0
    while i < K:
    	tmp = np.zeros((X.shape[0],))
    	num = 0
    	for j in range(0, P):
    		if W[i][j] == 1:
    			tmp = np.add(tmp , X[:,j])
    			num = num + 1
    	C[:,i] = tmp/num
        i += 1
    C_new = C
    flag = True
    max_iter = 1000000
    iter = 0
    while(flag):
    	W = np.zeros((K,P))
        i = 0
        while i < P:
    	    min_diff = 100000000
            min_index = -1
            j = 0
            while j < K:
                if np.linalg.norm(X[:,i] - C[:,j]) < min_diff:
                    min_diff = np.linalg.norm(X[:,i] - C[:,j])
                    min_index = j
                j += 1
            W[min_index][i] = 1
            i += 1
        C = np.zeros((X.shape[0],K))

        i = 0
        while i < K:
    	    tmp = np.zeros((X.shape[0],))
    	    num = 0
    	    for j in range(0,P):
    		    if W[i][j] == 1:
    			    tmp = np.add(tmp , X[:,j])
    			    num = num + 1

    	    C[:,i] = tmp/num
    	    if np.linalg.norm(C - C_new) < 0.00001:
    	    	flag = False
    	    if iter >= max_iter:
    	    	flag = False
            i += 1
    	iter += 1
    	C_new = C
        
    return C, W

def plot_results(X, C, W, C0):

    K = np.shape(C)[1]

    # plot original data
    fig = plt.figure(facecolor = 'white')
    ax1 = fig.add_subplot(121)
    plt.scatter(X[0,:],X[1,:], s = 50, facecolors = 'k')
    plt.title('original data')
    ax1.set_xlim(-.55, .55)
    ax1.set_ylim(-.55, .55)
    ax1.set_aspect('equal')

    plt.scatter(C0[0,0],C0[1,0],s = 100, marker=(5, 2), facecolors = 'b')
    plt.scatter(C0[0,1],C0[1,1],s = 100, marker=(5, 2), facecolors = 'r')

    # plot clustered data
    ax2 = fig.add_subplot(122)
    colors = ['b','r']

    for k in np.arange(0,K):
        ind = np.nonzero(W[k][:]==1)[0]
        plt.scatter(X[0,ind],X[1,ind],s = 50, facecolors = colors[k])
        plt.scatter(C[0,k],C[1,k], s = 100, marker=(5, 2), facecolors = colors[k])

    plt.title('clustered data')
    ax2.set_xlim(-.55, .55)
    ax2.set_ylim(-.55, .55)
    ax2.set_aspect('equal')

# load data
X = np.array(np.genfromtxt('Kmeans_demo_data.csv', delimiter=','))

C0 = np.array([[0,0],[0,.5]])   # initial centroid locations

# run K-means
K = np.shape(C0)[1]

C, W = your_K_means(X, K, C0)

# plot results
plot_results(X, C, W, C0)
plt.show()
