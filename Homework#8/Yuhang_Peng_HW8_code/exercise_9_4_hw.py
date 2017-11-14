# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

from __future__ import division
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import copy

# YOUR CODE GOES HERE - recommender systems via matrix completion
def matrix_complete(X, K):
    N = X.shape[0]
    P = X.shape[1]

    # C - N x K , W - K x P

    # initial point
    C0 = np.ones((N, K)) * 2.05
    W0 = np.ones((K, P)) * 1.04

    C = np.zeros((N, K))
    W = np.zeros((K, P))

    # X.shape = (100,200) C.shape = (100,5) W.shape = (5,200)
    i = 0
    while True:
        if i == P:
            break
        c = np.zeros((K,1))
        a = np.zeros((K,K))
        j = 0
        while j < N:
            if X[j][i] != 0:
                c = c + float(X[j][i]) * C0[j,:].reshape((K,1))
                a = a + np.dot(C0[j,:].T, C0[j,:])
            j += 1
        W[:,i] = np.dot(np.linalg.pinv(a),c.reshape((K,1))).reshape((K))
        i += 1

    i = 0
    while True:
        if i == N:
            break
        b = np.zeros((K,K))
        d = np.zeros((1,K))
        j = 0
        while j < P:
            if X[i][j] != 0:
                d = d + X[i][j] * W0[:,j].reshape((1,K))
                b = b + np.dot(W[:,j],W[:,j].T)
            j += 1
        C[i,:] = np.dot(d, np.linalg.pinv(b))
        i += 1
    C_new = np.ones((N, K))
    W_new = np.zeros((K, P))

    # By alternately solving these linear system until the values of C and W do not change very much
    alpha = 0.002
    while np.linalg.norm(C - C_new) > alpha and np.linalg.norm(W - W_new) > alpha:
        C_new = copy.deepcopy(C)
        W_new = copy.deepcopy(W)

        i = 0
        while True:
            if i == P:
                break
            c = np.zeros((K,1))
            a = np.zeros((K,K))
            j = 0
            while j < N:
                if X[j][i] != 0:
                    c = c + float(X[j][i]) * C_new[j,:].reshape((K,1))
                    a = a + np.dot(C_new[j,:].T,C_new[j,:])
                j += 1
            W[:,i] = np.dot(np.linalg.pinv(a), c.reshape((K,1))).reshape((K))
            i += 1

        i = 0
        while True:
            if i == N:
                break
            b = np.zeros((K,K))
            d = np.zeros((1,K))
            j = 0
            while j < P:
                if X[i][j] != 0:
                    d = d + X[i][j] * W_new[:,j].reshape((1,K))
                    b = b + np.dot(W[:,j],W[:,j].T)
                j += 1
            C[i,:] = np.dot(d, np.linalg.pinv(b))
            i += 1
    return C, W

def plot_results(X, X_corrupt, C, W):

    gaps_x = np.arange(0,np.shape(X)[1])
    gaps_y = np.arange(0,np.shape(X)[0])

    # plot original matrix
    fig = plt.figure(facecolor = 'white',figsize = (30,10))
    ax1 = fig.add_subplot(131)
    plt.imshow(X,cmap = 'hot',vmin=0, vmax=20)
    plt.title('original')

    # plot corrupted matrix
    ax2 = fig.add_subplot(132)
    plt.imshow(X_corrupt,cmap = 'hot',vmin=0, vmax=20)
    plt.title('corrupted')

    # plot reconstructed matrix
    ax3 = fig.add_subplot(133)
    recon = np.dot(C,W)
    plt.imshow(recon,cmap = 'hot',vmin=0, vmax=20)
    RMSE_mat = np.sqrt(np.linalg.norm(recon - X,'fro')/np.size(X))
    title = 'RMSE-ALS = ' + str(RMSE_mat)
    plt.title(title,fontsize=10)

# load in data
X = np.array(np.genfromtxt('recommender_demo_data_true_matrix.csv', delimiter=','))
X_corrupt = np.array(np.genfromtxt('recommender_demo_data_dissolved_matrix.csv', delimiter=','))

K = np.linalg.matrix_rank(X)

# run ALS for matrix completion
C, W = matrix_complete(X_corrupt, K)

# plot results
plot_results(X, X_corrupt, C, W)
plt.show()
