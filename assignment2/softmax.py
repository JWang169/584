# softmax: e^x / sum(e^x)
#        = e^(x-x(max))/sum(e^(x-max(x))
import numpy as np

# generate a matrix, given an input of N rows and D columns
def matrix(N=5, D=10):
	x = np.random.randint(100, size=(N, D))
	return x


# implement softmax(x+c)
def softmax(x):
	max_x = np.max(x, axis=1)
	max_x = np.reshape(max_x, (len(max_x), 1))
	xnorm = x - max_x
	e_xnorm = np.exp(xnorm)
	sum_e_xnorm = np.sum(e_xnorm, axis=1)
	sum_e_xnorm = np.reshape(sum_e_xnorm, (len(sum_e_xnorm), 1))
	soft_xnorm = e_xnorm/sum_e_xnorm
	print("softmax result:", soft_xnorm)
	return soft_xnorm


# test with original softmax(x)
def test(x):
	e_x = np.exp(x)
	sum_e_x = np.sum(e_x, axis=1)
	sum_e_x = np.reshape(sum_e_x, (len(sum_e_x), 1))
	soft_x = e_x/sum_e_x
	print("test result: ", soft_x)
	return soft_x

def main():
	x = matrix()
	res_softmax = softmax(x)
	res_test = test(x)


if __name__ == "__main__":
    main()

