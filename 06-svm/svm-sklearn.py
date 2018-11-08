# -*- coding: UTF-8 -*-
import numpy as np
from sklearn.svm import SVC
import np_util


def load_dataset1(filename):
    d = np.loadtxt(filename, delimiter=',', dtype=np.unicode)

    # change all str to int features
    l = list(range(9))
    l.pop(-2)
    data = np_util.str_feature_to_vector(d)
    # data = np_util.str_feature_to_int(d, l)

    # return label -> array map
    return np_util.split_by_label(data)


def sv():
    datas = load_dataset1('std_train')  # 加载训练集
    # datas = load_dataset1('data/post-operative.data')  # 加载训练集
    # dataArr, labelArr = load_dataset1('data/post-operative.data')  # 加载训练集
    d = np.insert(datas[0], 0, values=datas[2], axis=0)
    dArr = list(np.delete(d, (-1), axis=1))
    dataArr = []
    for da in dArr:
        dataArr.append(list(da))
    labelArr = list(d[:, -1])

    clf = SVC(C=200, kernel='rbf')
    clf.fit(np.array(dataArr), labelArr)

    cnt = 0
    for i in range(len(d)):
        classNumber = d[i][-1]
        vectorUnderTest = d[i][0:-1].reshape((1, -1))
        classifierResult = clf.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if classifierResult != classNumber:
            cnt += 1
    print(cnt, len(d), cnt / len(d))


if __name__ == '__main__':
    sv()
