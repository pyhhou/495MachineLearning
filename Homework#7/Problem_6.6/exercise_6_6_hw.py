import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import math
import csv
from operator import itemgetter

def load_data(filename):
	data = np.array(np.genfromtxt(filename, delimiter=','))
	x = data[:,0]
	y = data[:,1]
	label = data[:,2]
	x = np.reshape(x,(np.size(x),1))
	y = np.reshape(y,(np.size(y),1))
	label = np.reshape(label,(np.size(label),1))
	return x, y, label

def divide_space(densities):
	d_x = np.linspace(0, 10, densities)
	d_y = np.linspace(0, 10, densities)

	return d_x, d_y

def distance_calculate(x, y, label, d_x, d_y, k):
	results = []
	for i in d_x:
		for j in d_y:
			distance = []
			for cur in range(len(x)):
				distance.append([math.sqrt((x[cur] - i) ** 2 + (y[cur] - j) ** 2), label[cur]])
			distance = sorted(distance, key = itemgetter(0))

			result = classfication(distance, k)
			results.append(result)
		# distances.append([distance, cur])
	return results

def classfication(distance, k):
	total = 0
	result = 0
	for i in range(k):
		total += distance[i][1]
	avg = total / k
	if avg > 0.5:
		result = 1
	else:
		result = -1

	return result

# plot points
def plot_points(x,y,label):

    for i in range(len(label)):
    	if label[i][0] == 0:
    		plt.plot(x[i], y[i], 'ro')
    	else:
    		plt.plot(x[i], y[i], 'bo')
    	plt.hold(True)

#plot classification division
def plot_space(d_x, d_y, results):
    index = 0
    for x in d_x:
		for y in d_y:
			if results[index] == -1:
				plt.plot(x, y, 'go')
			index += 1
			plt.hold(True)

k = 10
densities = 100

x, y, label = load_data('knn_data.csv')

# print x
# print y
# print label

d_x, d_y = divide_space(densities)

# print d_x
# print d_y

results = distance_calculate(x, y, label, d_x, d_y, k)

print results

# plot resulting fit
fig = plt.figure(facecolor = 'white',figsize = (4,4))
plot_space(d_x, d_y, results)
plot_points(x, y, label)
plt.show()
