from pylab import *
import matplotlib.pyplot as plt
import csv

def read_data(filename):
    f = open(filename)
    csv_f = csv.reader(f)
    x = []
    y = []
    for row in csv_f:
    	temp = array([1, float(row[0])])
    	temp.shape = (2, 1)
        x.append(temp)
        y.append(float(row[1]))
    print(y)
    return x, y

def basic_linear_regression(x,y):
    A = array([0.,0.,0.,0.])
    A.shape = (2,2)
    for i in x:
    	A += dot(i,i.T)
    A_inv = pinv(A)

    b = array([0.,0.])
    b.shape = (2,1)
    for i in range(len(x)):
    	b += y[i] * x[i]
    w = dot(A_inv,b)
    return w

def main():
    x, y = read_data('bacteria_data.csv')                 # plot objective function
    y_p = []
    for i in y:
        temp = np.log((i) / (1 - i))
        y_p.append(temp)
    w = basic_linear_regression(x,y_p)
    _x = []
    _y = []
    _b = w[0][0]
    _w = w[1][0]
    print(w)
    for i in range(len(x)):
    	_x.append(x[i][1])
    	temp = np.e**(_b + x[i][1] * _w) / (1 + np.e**(_b + x[i][1] * _w))
    	_y.append(temp)
    # print(w)
    # predict = _b + _w * 2050
    # print(predict)

    plt.plot(_x,y,'b.')
    plt.plot(_x,_y,'r')
    plt.title('Bacteria Data')
    plt.xlabel('t/hours')
    plt.ylabel('concentration')
    show()
main()