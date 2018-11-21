# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet():
    d = np.loadtxt('data/movie.txt', delimiter='\t')
    rating = d[:, 0]
    gross = d[:, 2]
    data = d[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    return rating, gross, data


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.000000001  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.zeros((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights - alpha * dataMatrix.transpose() * error
    return weights.getA()  # 将矩阵转换为数组，返回权重数组


def computeCost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))


def bgd(x, y, theta, iters=500, alpha=0.0001):
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iters)

    for i in range(iters):
        error = (x * theta.T) - y

        for j in range(10):
            term = np.multiply(error, x[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(x)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(x, y, theta)

    return theta, cost


if __name__ == '__main__':
    rating, gross, data = loadDataSet()

    # weights = gradAscent(data, rating)
    t = np.matrix(np.zeros((1, 10)))
    data = (data - data.mean()) / data.std()
    theta, cost = bgd(np.matrix(data), np.matrix(rating), t)
    # print(theta, cost)
    cnt = 0
    ecnt = 0
    mcnt = 0
    for i, item in enumerate(data):
        pd = sum(np.array(theta)[0] * np.array(item))
        print(rating[i], pd)
        if abs(rating[i] - pd) <= 0.25:
            ecnt += 1
        if abs(rating[i] - pd) <= 1:
            cnt += 1
        if abs(rating[i] - pd) <= 2:
            mcnt += 1

    print('accuracy1: ', ecnt / len(data))
    print('accuracy2: ', cnt / len(data))
    print('accuracy3: ', mcnt / len(data))
