# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2017-03-25
"""


def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加,sum(0)列相加,sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方,计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


def letter_recognition():
    # labels, training set
    labels = []
    data_size = 20000
    train_size = 19500
    test_size = data_size - train_size
    data = np.zeros((data_size, 16))
    with open('data/letter-recognition.data') as f:
        for i, line in enumerate(f, 1):
            labels.append(line[0])
            data[i - 1, :] = np.fromstring(line.split(',', 1)[1], dtype=int, sep=',')

    t = np.vsplit(data, np.array([train_size]))
    train_data = t[0]
    test_data = t[1]
    err_count = 0

    for i, line in enumerate(test_data):
        result = classify0(line, train_data, labels, 3)
        if result != labels[train_size + i]:
            err_count += 1

    print("总共错了%d个数据\n错误率为%f%%" % (err_count, err_count / test_size))


"""
函数说明:main函数

Parameters:
	无
Returns:
	无

Modify:
	2017-03-25
"""
if __name__ == '__main__':
    letter_recognition()
