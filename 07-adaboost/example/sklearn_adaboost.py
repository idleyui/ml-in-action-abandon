# -*-coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
# from sklearn.
from sklearn.ensemble import AdaBoostRegressor
import np_util

"""
Author:
	Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
	2017-10-11
"""


def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat


def load_data(filename):
    d = np.loadtxt(filename, delimiter=';', dtype=np.unicode)
    d = np.delete(d, (0), axis=0)
    for item in d:
        item[-2] = item[-2][1:-1]
        item[-3] = item[-3][1:-1]
        item[-1] = 1 if int(item[-1]) > 10 else -1
    data = np_util.str_feature_to_int(d, [0, 1, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22])
    return np.delete(data, (-1), axis=1), data[:,-1]


if __name__ == '__main__':
    # dataArr, classLabels = loadDataSet('horseColicTraining2.txt')
    dataArr, classLabels = load_data('data/student-mat.txt')
    testArr, testLabelArr = load_data('data/student-por.txt')
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=10)
    bdt.fit(dataArr, classLabels)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))

    # regression
    # ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2), n_estimators=100, random_state=1)
    # ada.fit(dataArr, classLabels)
    # predictions = ada.predict(dataArr)
    # l = predictions - classLabels
    # print(sum([0 if abs(item) < 2.5 else 1 for item in l]))
    # print("ok")

