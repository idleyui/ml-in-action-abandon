import pandas as pd
import numpy as np

# data2.head()
# data2 = (data2 - data2.mean()) / data2.std()
# data2.head()


def computeCost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

if __name__ == '__main__':
    # add ones column
    path = 'data\ex1data2.txt'
    data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
    data2.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data2.shape[1]
    X2 = data2.iloc[:, 0:cols - 1]
    y2 = data2.iloc[:, cols - 1:cols]

    # convert to matrices and initialize theta
    X2 = np.matrix(X2.values)
    y2 = np.matrix(y2.values)
    theta2 = np.matrix(np.array([0, 0, 0]))
    alpha = 0.01
    iters = 1000
    # perform linear regression on the data set
    g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

    # get the cost (error) of the model
    computeCost(X2, y2, g2)
