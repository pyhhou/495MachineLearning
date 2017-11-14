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
    x, y = read_data('student_debt.csv')                 # plot objective function
    w = basic_linear_regression(x,y)
    _x = []
    _y = []
    _b = w[0][0]
    _w = w[1][0]
    for i in range(len(x)):
    	_x.append(x[i][1])
    	temp = x[i][1] * _w + _b
    	_y.append(temp)
    print(w)
    predict = _b + _w * 2050
    print(predict)

    plt.plot(_x,y,'g.')
    plt.plot(_x,_y,'b')
    plt.title('Student Debt Data')
    plt.xlabel('Years')
    plt.ylabel('Total Debt')
    show()
main()