from __future__ import print_function
import numpy as np
import csv

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def checkSize(w, X, y):
    # w and y are column vector, shape [N, 1] not [N,]
    # X is a matrix where rows are data sample
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == w.shape[0]
    assert len(y.shape) == 2
    assert len(w.shape) == 2
    assert w.shape[1] == 1
    assert y.shape[1] == 1

def compactNotation(X):
    return np.hstack([np.ones([X.shape[0], 1]), X])

def readData(path):
    """
    Read data from path (either path to MNIST train or test)
    return X in compact notation (has one appended)
    return Y in with shape [10000,1] and starts from 0 instead of 1
    """
    reader = csv.reader(open(path,'r'), delimiter = ',')
    d = list(reader)

    data = np.array(d).astype('float')
    X = data[:,:-1]
    X = compactNotation(X)
    y = data[:,-1] - 1
    y.shape = (len(y),1)

    return X,y

def softmaxGrad(w, X, y):
    checkSize(w, X, y)
    ### RETURN GRADIENT
    r = -sigmoid(-y*np.dot(X,w))*y
    grad = np.dot(X.T,r)

    return grad

def accuracy(OVA, X, y):
    """
    Calculate accuracy using matrix operations!
    """
    temp = np.dot(X,OVA)
    res = np.argmax(temp,axis=1)
    res.shape=(len(y),1)

    return np.sum(y==res) / len(y)

def gradientDescent(grad, w0, *args, **kwargs):
    max_iter = 5000
    alpha = 0.001
    eps = 10**(-5)

    w = w0
    iter = 0
    while True:
            # print(iter)
            gradient = grad(w, *args)
            w = w - alpha * gradient

            if iter > max_iter or np.linalg.norm(gradient) < eps:
                    break

            if iter  % 1000 == 1:
                    print("Iter %d " % iter)

            iter += 1
    return w

def oneVersusAll(Y, value):
    """
    generate label Yout,
    where Y == value then Yout would be 1
    otherwise Yout would be -1
    """
    Yout = []
    for i in range(len(Y)):
        if Y[i][0] == value:
            Yout.append(1)
        else:
            Yout.append(-1)
    Yout=np.array(Yout)
    Yout.shape = (len(Y),1)
    return Yout

if __name__=="__main__":
    trainX, trainY = readData('MNIST_data/MNIST_train_data.csv')
    # training individual classifier
    Nfeature = trainX.shape[1]
    Nclass = 10
    OVA = np.zeros((Nfeature, Nclass))
    for i in range(Nclass):
            print("Training for class " + str(i))
            w0 = np.random.rand(Nfeature,1)
            OVA[:, i:i+1] = gradientDescent(softmaxGrad, w0, trainX, oneVersusAll(trainY, i))


    print("Accuracy for training set is: %f" % accuracy(OVA, trainX, trainY))

    testX, testY = readData('MNIST_data/MNIST_test_data.csv')
    print("Accuracy for test set is: %f" % accuracy(OVA, testX, testY))
