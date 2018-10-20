import matplotlib.pyplot as plt
import numpy as np
import random


class SMO:
    def __init__(self, data, labels, C=0.6, toler=0.001, max_iter=40):
        self.data = data
        self.labels = labels
        self.C = C
        self.toler = toler
        self.row, self.column = np.shape(data)
        self.alphas = np.zeros((self.row, 1))
        self.b = 0
        self.eCache = np.zeros((self.row, 2))
        self.max_iter = max_iter

    def error(self, i):
        return float((self.alphas * self.labels).T @ (self.data @ self.data[i, :].T)) + self.b \
               - float(self.labels[i])

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


def load_dataset(filename):
    # d = np.loadtxt(filename, delimiter=',')
    d = np.loadtxt(filename)
    return d[:, [0, 1, 2]], d[:, [-1]]


def showClassifer():
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(data)):
        if labels[i] > 0:
            data_plus.append(data[i])
        else:
            data_minus.append(data[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)  # 正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)  # 负样本散点图
    # 绘制直线
    x1 = max(data[:, 0])
    x2 = min(data[:, 0])
    a1, a2 = w
    # b = float(b)
    b = float(bn)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = data[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()


if __name__ == '__main__':
    # data, labels = load_dataset('data/post-operative.data')
    data, labels = load_dataset('data/testSet1.txt')
    smo = SMO(data, labels)
    bn, alphas = smo.smoP()
    w = smo.calc_ws()
    # showClassifer(data, w, b)
    # showClassifer()
