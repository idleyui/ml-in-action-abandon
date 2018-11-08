# -*- coding: UTF-8 -*-
import numpy as np
from collections import Counter


def classify0(test_data, data_set, labels, k):
    diff = np.tile(test_data, (data_set.shape[0], 1)) - data_set
    distances = (diff ** 2).sum(axis=1) ** 0.5
    k_labels = labels[distances.argsort()[:k]]
    return Counter(k_labels).most_common()[0][0]


def letter_recognition():
    data_size = 20000
    train_size_list = [10000, 15000, 19000]
    for train_size in train_size_list:
        test_size = data_size - train_size
        for k in range(1, 10):
            d = np.loadtxt('data/letter-recognition.data', delimiter=',', dtype='<U3')
            labels = d[:, 0]
            data = np.delete(d, 0, axis=1).astype(np.float)
            train_data, test_data = np.vsplit(data, np.array([train_size]))

            re = [classify0(item, train_data, labels, k) == labels[train_size + i] for i, item in enumerate(test_data)]
            right = Counter(re).most_common()[0][1]

            print("train with %d data item and k=%d, test with %d data item, %d item is right, accuracy is %.2f%%"
                  % (train_size, k, test_size, right, (right / test_size) * 100))


if __name__ == '__main__':
    letter_recognition()
