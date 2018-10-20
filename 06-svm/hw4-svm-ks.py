import matplotlib.pyplot as plt
import numpy as np
import random

from mynp import np_util
from collections import Counter


class SMO:
    def __init__(self, data, labels, C=0.6, toler=0.001, max_iter=40, kTup=('lin',0)):
        self.data = data
        self.labels = labels
        self.C = C
        self.toler = toler
        self.row, self.column = np.shape(data)
        self.alphas = np.zeros((self.row, 1))
        self.b = 0
        self.eCache = np.zeros((self.row, 2))
        self.max_iter = max_iter
        self.K = np.zeros(((self.row, self.row)))
        for i in range(self.row):
            self.K[:, i] = self.kernel_trans(self.data, self.data[i, :], kTup)

    def error(self, i):
        # return float((self.alphas * self.labels).T @ (self.data @ self.data[i, :].T)) + self.b \
        rt = float((self.alphas * self.labels).T @ self.K[:, i] + self.b) - float(self.labels[i])
        return rt

    def selectJrand(self, i):
        j = i
        while j == i: j = int(random.uniform(0, self.row))
        return j

    def selectJ(self, i, ei):
        maxK = -1
        maxDeltaE = 0
        ej = 0
        self.eCache[i] = [1, ei]
        valid_ecaches = np.nonzero(self.eCache[:, 0])[0]
        if len(valid_ecaches) > 1:
            for k in valid_ecaches:
                if k == i: continue
                ek = self.error(k)
                deltaE = abs(ei - ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    ej = ek
            return maxK, ej
        else:
            j = self.selectJrand(i)
            ej = self.error(j)
        return j, ej

    def updateError(self, i):
        self.eCache[i] = [1, self.error(i)]

    def clip_alpha(self, aj, h, l):
        if aj > h: aj = h
        if l > aj: aj = l
        return aj

    def innerL(self, i):
        # step 1: calc error ei
        Ei = self.error(i)

        if (self.labels[i] * Ei < -self.toler and self.alphas[i] < self.C) \
                or (self.labels[i] * Ei > self.toler and self.alphas[i] > 0):
            # step 1: calc error ej
            j, Ej = self.selectJ(i, Ei)
            # Ej = self.error(j)
            old_i, old_j = self.alphas[i].copy(), self.alphas[j].copy()
            # step 2: calc boundary l and h
            if self.labels[i] != self.labels[j]:
                l = max(0, self.alphas[j] - self.alphas[i])
                h = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                l = max(0, self.alphas[j] + self.alphas[i] - self.C)
                h = min(self.C, self.alphas[j] + self.alphas[i])
            if l == h: return 0
            # step 3: calc eta
            eta = 2.0 * self.data[i, :] @ self.data[j, :].T - self.data[i, :] @ self.data[i, :].T \
                  - self.data[j, :] @ self.data[j, :].T
            if eta >= 0: return 0
            # step 4: update alphas[j] and clip it
            self.alphas[j] -= self.labels[j] * (Ei - Ej) / eta
            self.alphas[j] = self.clip_alpha(self.alphas[j], h, l)
            self.updateError(j)
            if abs(self.alphas[j] - old_j) < 0.0001: return 0
            # step 5: update alphas[i]
            self.alphas[i] += self.labels[j] * self.labels[i] * (old_j - self.alphas[j])
            self.updateError(i)
            # step 6: update b1 and b2
            b1 = self.b - Ei - self.labels[i] * (self.alphas[i] - old_i) * self.data[i, :] @ self.data[i, :].T \
                 - self.labels[j] * (self.alphas[j] - old_j) * self.data[i, :] @ self.data[j, :].T
            b2 = self.b - Ej - self.labels[i] * (self.alphas[i] - old_i) * self.data[i, :] @ self.data[j, :].T \
                 - self.labels[j] * (self.alphas[j] - old_j) * self.data[j, :] @ self.data[j, :].T
            if 0 < self.alphas[i] and self.C > self.alphas[i]:
                self.b = b1
            elif 0 < self.alphas[j] and self.C > self.alphas[j]:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2
            return 1
        else:
            return 0

    def smoP(self):
        iter = 0
        entire = True
        updatecnt = 0
        while iter < self.max_iter and (updatecnt > 0 or entire):
            updatecnt = 0
            if entire:
                for i in range(self.row):
                    updatecnt += self.innerL(i)
            else:
                nonBound = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in nonBound:
                    updatecnt += self.innerL(i)
            iter += 1
            if entire:
                entire = False
            elif updatecnt == 0:
                entire = True
            print("iter: ", iter)
        return self.b, self.alphas

    def calc_ws(self):
        w = np.zeros((self.column, 1))
        for i in range(self.row):
            w += ((self.alphas[i].reshape((1, 1)) * self.labels[i].reshape(1, 1)) * self.data[i, :]).T
        return w

    def kernel_trans(self, X, A, kTup):
        m, n = np.shape(X)
        K = np.zeros((m, 1))
        if kTup[0] == 'lin':
            K = X @ A.T  # 线性核函数,只进行内积。
        elif kTup[0] == 'rbf':  # 高斯核函数,根据高斯核函数公式进行计算
            for j in range(m):
                deltaRow = X[j, :] - A
                K[j] = deltaRow @ deltaRow.T
            K = np.exp(K / (-1 * kTup[1] ** 2))  # 计算高斯核K
        else:
            print(kTup)
            raise NameError('核函数无法识别')
        return K[0]  # 返回计算的核K

    def classify(self, index, k1=1.3):
        sv_index = np.nonzero(self.alphas > 0)[0]
        svs = self.data[sv_index]
        sv_label = self.labels[sv_index]
        print("sv num:", len(svs))

        kernel_eval = self.kernel_trans(svs, self.data[index, :], ('rbf', k1))
        predict = kernel_eval.T @ (sv_label * self.alphas[sv_index]) + self.b
        return predict


def load_dataset(filename):
    d = np.loadtxt(filename, delimiter=',', dtype=np.unicode)

    # change all str to int features
    l = list(range(9))
    l.pop(-2)
    data = np_util.str_feature_to_int(d, l)

    # return label -> array map
    return np_util.split_by_label(data)


if __name__ == '__main__':
    datas = load_dataset('data/post-operative.data')

    svms = []
    for i in range(len(datas)):
        for j in range(i, len(datas)):
            d = np.insert(datas[i], 0, values=datas[j], axis=0)
            svms.append(SMO(np.delete(d, (-1), axis=1), d[:, -1]))

    for s in svms:
        s.smoP()

    for i in range(len(datas)):
        l = [s.classify(i) for s in svms]
        print(datas[i][-1], Counter(l).most_common())

    # todo predict i
