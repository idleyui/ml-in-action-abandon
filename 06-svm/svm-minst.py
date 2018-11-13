# -*-coding:utf-8 -*-
import numpy as np
import random
import pandas as pd
import pickle
from collections import Counter
import os


class optStruct:

    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn  # 数据矩阵
        self.labelMat = classLabels  # 数据标签
        self.C = C  # 松弛变量
        self.tol = toler  # 容错率
        self.m = np.shape(dataMatIn)[0]  # 数据矩阵行数
        self.alphas = np.mat(np.zeros((self.m, 1)))  # 根据矩阵行数初始化alpha参数为0
        self.b = 0  # 初始化b参数为0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 根据矩阵行数初始化虎误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值。
        self.K = np.mat(np.zeros((self.m, self.m)))  # 初始化核K
        print('calc kernel', self.m)
        for i in range(self.m):  # 计算所有数据的核K,
            if i % 100 == 0:
                print(i)
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
        print('finish')


def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核函数,只进行内积。
    elif kTup[0] == 'rbf':  # 高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))  # 计算高斯核K
    else:
        raise NameError('核函数无法识别')
    return K  # 返回计算的核K


def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJrand(i, m):
    j = i  # 选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def selectJ(i, oS, Ei):
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0  # 初始化
    oS.eCache[i] = [1, Ei]  # 根据Ei更新误差缓存
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:  # 遍历,找到最大的Ek
            if k == i: continue  # 不计算i,浪费时间
            Ek = calcEk(oS, k)  # 计算Ek
            deltaE = abs(Ei - Ek)  # 计算|Ei-Ek|
            if (deltaE > maxDeltaE):  # 找到maxDeltaE
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej  # 返回maxK,Ej
    else:  # 没有不为0的误差
        j = selectJrand(i, oS.m)  # 随机选择alpha_j的索引值
        Ej = calcEk(oS, j)  # 计算Ej
    return j, Ej  # j,Ej


def updateEk(oS, k):
    Ek = calcEk(oS, k)  # 计算Ek
    oS.eCache[k] = [1, Ek]  # 更新误差缓存


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def innerL(i, oS):
    # 步骤1：计算误差Ei
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率。
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        # 步骤2：计算上下界L和H
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        # 步骤3：计算eta
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
                oS.alphas[j] - alphaJold) * oS.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def split_by_label(arr: np.ndarray, index: int):
    uniq_label = np.unique(arr[:, index])
    return [arr[np.where(arr[:, index] == label)] for label in uniq_label]


def load_data(filename):
    d = pd.read_csv(filename, delimiter=',').values

    # return label -> array map
    return split_by_label(d, 0)


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet:  # 遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 使用优化的SMO算法
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):  # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        print("迭代次数: %d" % iter)
    return oS.b, oS.alphas  # 返回SMO算法计算的b和alphas


def one2rest():
    # load train and test data
    data = pd.read_csv('std_train', delimiter=',').values
    test_data = pd.read_csv('std_test', delimiter=',').values
    svms = []
    # k1 = 1.3
    # k1 = 0.05
    k1 = 5.5

    # check if there are saved svms
    cal = True
    if os.path.exists('svms1'):
        with open('svms1', 'rb') as f:
            svms = pickle.load(f)
        cal = False

    # make 9 one to rest svms
    dataArr = []
    labelArr = []
    for i in range(9):
        data_i = np.delete(data[np.where(data[:, 0] == i)], (0), axis=1)
        label_i = np.full((len(data_i, ),), 1)
        data_rest = np.delete(data[np.where(data[:, 0] > i)], (0), axis=1)
        label_rest = np.full((len(data_rest),), -1)
        d1 = np.insert(data_i, 0, values=data_rest, axis=0)
        data_train = [list(item) for item in d1]
        l1 = np.insert(label_i, 0, values=label_rest, axis=0)
        label_train = list(l1)
        if cal:
            svms.append(smoP(data_train, label_train, 150, 0.0001, 100, ('rbf', k1)))
        print('svms:', len(svms[i][1]))
        dataArr.append(data_train)
        labelArr.append(label_train)

    if cal:
        with open('svms1', 'wb') as f:
            pickle.dump(svms, f)

    # test
    label = test_data[:, 0]
    data_test = np.delete(test_data, (0), axis=1)
    # data_test = [list(item) for item in d]
    p = []

    # get support vectors
    dataMats = [np.mat(dataArr[i]) for i in range(9)]
    labelMats = [np.mat(labelArr[i]).transpose() for i in range(9)]
    svInds = [np.nonzero(svms[i][1].A > 0)[0] for i in range(9)]
    sVss = [dataMats[i][svInds[i]] for i in range(9)]
    labelSVs = [labelMats[i][svInds[i]] for i in range(9)]
    for svs in sVss:
        print('support vector num:', np.shape(svs)[0])

    for index in range(len(data_test)):
        print(index)
        for i in range(9):
            svInd = svInds[i]
            sVs = sVss[i]
            labelSV = labelSVs[i]
            b = svms[i][0]
            alphas = svms[i][1]

            kernelEval = kernelTrans(sVs, data_test[index, :], ('rbf', k1))  # 计算各个点的核
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + float(b)  # 根据支持向量的点，计算超平面，返回预测结果
            if float(predict) > 0:
                p.append(i)
                break
            elif i == 8:
                p.append(9)
        if index % 10 == 0:
            print(sum([1 if p[j] == label[j] else 0 for j in range(index)]))

    error_cnt = 0
    for i in range(len(p)):
        if p[i] != label[i]:
            error_cnt += 1

    print('use 5000 number to test')
    print('error cnt:', error_cnt)
    print('error rate: %f' % error_cnt / len(p))


def std(data):
    label = data[:, 0]
    d = np.delete(data, (0), axis=1)
    d[d > 0] = 1
    # d = d/255
    return np.insert(d, 0, values=label, axis=1)


def preprocess():
    data = pd.read_csv('16.MNIST.train.csv').values
    train_d = np.vsplit(data, [5000])[0]
    test_d = np.vsplit(data, [37000])[1]

    std_train_d = std(train_d)
    std_test_d = std(test_d)

    np.savetxt('std_train', std_train_d, delimiter=',', fmt='%i')
    np.savetxt('std_test', std_test_d, delimiter=',', fmt='%i')
    # np.savetxt('std_train', std_train_d, delimiter=',')
    # np.savetxt('std_test', std_test_d, delimiter=',')


if __name__ == '__main__':
    # tesRbf()
    # get_test()
    # d = pd.read_csv('16.MNIST.train.csv').values
    # np.savetxt('mnist_test', np.vsplit(d, [37000])[1], delimiter=',', fmt='%i')
    # preprocess()
    one2rest()
