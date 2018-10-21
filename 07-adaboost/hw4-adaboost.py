# -*-coding:utf-8 -*-
import numpy as np
import np_util


def load_data(filename):
    d = np.loadtxt(filename, delimiter=';', dtype=np.unicode)
    d = np.delete(d, (0), axis=0)
    for item in d:
        item[-2] = item[-2][1:-1]
        item[-3] = item[-3][1:-1]
        item[-1] = 1 if int(item[-1]) > 10 else -1
    data = np_util.str_feature_to_int(d, [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22])
    return np.delete(data, (-1), axis=1), data[:, -1]


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max()  # 找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权重
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 构建单层决策树
        # print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(bestStump)  # 存储单层决策树
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst  # 计算类别估计累计值
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))  # 计算误差
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0: break  # 误差为0，退出循环
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):  # 遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    # print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    dataArr, LabelArr = load_data('data/student-mat.txt')
    # dataArr, LabelArr = load_data('data/student-por.txt')
    da = []
    for d in dataArr:
        da.append(list(d))
    dataArr = da
    LabelArr = list(LabelArr)

    testArr, testLabelArr = load_data('data/student-por.txt')
    # testArr, testLabelArr = load_data('data/student-mat.txt')
    ta = []
    for d in testArr:
        ta.append(list(d))
    testArr = ta
    testLabelArr = list(testLabelArr)

    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    # test token
    # print(weakClassArr)
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
