import numpy as np
from numpy.linalg import norm
def knn(vector, matrix, k=10):

    """ k-nearest neighbors algorithm.
    Arguments:
    vector -- C dimension vector
    matrix -- R * C dimension numpy matrix 
    k -- integer

    Return:
    nearestIdx -- a vector of rows indices of the k-nearest neighbors in the matrix

    """
    normVec = norm(vector) # sqrt(sum(ai^2)), a is the vector
    distVec = np.zeros(matrix.shape[0]) # save all cosine distance between the input vector and each row in the matrix 
    
    for i in range(matrix.shape[0]):
        # computing cosine similarity as a distance metric
        prod = np.dot(vector, matrix[i, :])
        normRow = norm(matrix[i, :])

        distVec[i] = prod / normVec / normRow
    # print(distVec)
    idx = np.argsort(distVec)
    nearestIdx = idx[-k:]

    return nearestIdx

def test_knn():
    indices = knn(np.array([0.2,0.5]), np.array([[0.1,0.5],[-0.5,0.1],[0,1],[2,-2],[4,4],[0.2,0.5], [3,3]]), k=2)
    print(indices)

if __name__ == "__main__":
    test_knn()
