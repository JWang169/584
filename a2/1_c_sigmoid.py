# sigmoid function: 
# sigmoid(x) = 1/(1+e^(-x))
# gradient of sigmoid(x) = sigmoid(x) * (1-sigmoid(x)) 

import numpy as np

def sigmoid(x):
	s = 1 / (1 + np.exp(-x))
	return s

def gradient(x):
	ds = np.exp(-x) / (1 + np.exp(-x) ** 2)
	return ds


def test(x):
	t = sigmoid(x) * (1 - sigmoid(x))
	return t

x = 10
x = float(input("Please input a number"))
s = sigmoid(x)
ds = gradient(x)
t = test(x)

print('The gradient of sigmoid function is', ds, '\n the test result is', t)