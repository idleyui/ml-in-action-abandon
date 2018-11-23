import numpy as np
import pandas as pd
from scipy import stats


def load(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # remove mean
    covMat = np.cov(meanRemoved, rowvar=0)
    eig_val, eig_vector = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eig_val)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eig_vector[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print('len:', topNfeat)
    # print('max var:', eig_vector[:,eigValInd[-1]])
    print('max var:', eig_val[0])
    return lowDDataMat, reconMat


def main():
    data = np.loadtxt('data/imports-85.data', delimiter=',', dtype=np.unicode)
    l = list(range(2, 9)) + [14, 15, 17]
    data[data == '?'] = 0
    for i in l:
        data[:, i], uniques = pd.factorize(data[:, i])
    data = data.astype(np.float)
    # for i in range(np.shape(data)[1]):
    #     data[:, i] = stats.zscore(data[:, i])
    # print(data.size)
    d = 25
    while d > 3:
        data, re = pca(data, d)
        d -= 1
    # d, re = pca(data, 25)
    # print(len(d))


def tp():
    arr = np.array([[-1, -2], [-1, 0], [0, 0], [2, 1], [0, 1]])
    d, r = pca(arr, 1)
    print(d)


if __name__ == '__main__':
    # main()
    # tp()
    data = load('data/misc.txt')
    print(data)
