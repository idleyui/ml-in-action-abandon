# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:梯度上升算法测试函数

求函数f(x) = -x^2 + 4x的极大值

Parameters:
	无
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""


def Gradient_Ascent_test():
    def f_prime(x_old):  # f(x)的导数
        return -2 * x_old + 4

    x_old = -1  # 初始值，给一个小于x_new的值
    x_new = 0  # 梯度上升算法初始值，即从(0,0)开始
    alpha = 0.01  # 步长，也就是学习速率，控制更新的幅度
    presision = 0.00000001  # 精度，也就是更新阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)  # 上面提到的公式
    print(x_new)  # 打印最终求解的极值近似值


def loadDataSet():
    d = np.loadtxt('data/movie.txt', delimiter='\t')
    rating = d[:, 0]
    gross = d[:, 2]
    data = d[:, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    return rating, gross, data


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明:梯度上升算法

Parameters:
	dataMatIn - 数据集
	classLabels - 数据标签
Returns:
	weights.getA() - 求得的权重数组(最优参数)
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Zhihu:
	https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-08-28
"""


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
    data = (data - data.mean())/data.std()
    theta, cost = bgd(np.matrix(data),np.matrix(rating), t)
    print(theta, cost)
    cnt = 0
    for i, item in enumerate(data):
        pd = sum(np.array(theta)[0] * np.array(item))
        print(rating[i], pd)
        if abs(rating[i] - pd) <=1:
            cnt+=1

    print(cnt/len(data))
