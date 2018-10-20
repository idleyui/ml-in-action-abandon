import matplotlib.pyplot as plt
import numpy as np
import random


def load_dataset(filename):
    d = np.loadtxt(filename, delimiter='\t')
    return d[:, [0, 1]], d[:, [-1]]


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if l > aj:
        aj = l
    return aj


def simple_SMO(data, labels, C, toler, max_iter=40):
    b = 0
    row, column = np.shape(data)
    alphas = np.zeros((row, 1))

    def error(num):
        fx = float((alphas * labels).T @ (data @ data[num].T)) + b
        return fx - float(labels[num])

    for iter in range(max_iter):
        updatecnt = 0
        for i in range(row):
            # step 1: calc error ei
            Ei = error(i)

            if (labels[i] * Ei < -toler and alphas[i] < C) \
                    or (labels[i] * Ei > toler and alphas[i] > 0):
                # step 1: calc error ej
                j = select_j_rand(i, row)
                Ej = error(j)
                old_i, old_j = alphas[i].copy(), alphas[j].copy()
                # step 2: calc boundary l and h
                if labels[i] != labels[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(C, C + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - C)
                    h = min(C, alphas[j] + alphas[i])
                if l == h: continue
                # step 3: calc eta
                eta = 2.0 * data[i, :] @ data[j, :].T - data[i, :] @ data[i, :].T \
                      - data[j, :] @ data[j, :].T
                if eta >= 0: continue
                # step 4: update alphas[j] and clip it
                alphas[j] -= labels[j] * (Ei - Ej) / eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if abs(alphas[j] - old_j) < 0.0001: continue
                # step 5: update alphas[i]
                alphas[i] += labels[j] * labels[i] * (old_j - alphas[j])
                # step 6: update b1 and b2
                b1 = b - Ei - labels[i] * (alphas[i] - old_i) * data[i, :] @ data[i, :].T \
                     - labels[j] * (alphas[j] - old_j) * data[i, :] @ data[j, :].T
                b2 = b - Ej - labels[i] * (alphas[i] - old_i) * data[i, :] @ data[j, :].T \
                     - labels[j] * (alphas[j] - old_j) * data[j, :] @ data[j, :].T
                if 0 < alphas[i] and C > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                updatecnt += 1
                print("iter %d time, sample: %d, update count:%d" % (iter, i, updatecnt))
        if updatecnt == 0:
            iter += 1
        else:
            iter_num = 0
        print("iter time: %d" % iter)
    return b, alphas


def get_w(data, labels, alphas):
    w = np.dot((np.tile(labels.reshape(1, -1).T, (1, 2)) * data).T, alphas)
    return w.tolist()


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
    data, labels = load_dataset('data/testSet.txt')
    bn, alphas = simple_SMO(data, labels, 0.6, 0.001)
    w = get_w(data, labels, alphas)
    # showClassifer(data, w, b)
    showClassifer()
