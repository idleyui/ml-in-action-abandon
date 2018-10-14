from collections import Counter
from functools import reduce
from math import log

import numpy as np

train_num = 12000


def load_data():
    d = np.loadtxt('data/nursery.data', delimiter=',', dtype=np.unicode)
    train, test = np.vsplit(np.delete(d, -1, axis=1), np.array([train_num]))
    mp = {item: i for i, item in enumerate(list(np.unique(d[:, -1])))}
    classes = np.array([mp[item] for item in d[:, -1]])
    return np.delete(d, -1, axis=1), train, test, classes


def trainNB0(data, classes):
    class2cnt = Counter(classes)
    c_len = len(np.unique(classes))
    pNum = np.zeros((c_len, len(data[0])))
    pDenom = np.zeros((1, c_len))
    for i, doc in enumerate(data):
        pNum[classes[i]] += doc
        pDenom[0][classes[i]] += sum(doc)
    vector = [pNum[i] / pDenom[0][i] for i in range(len(pNum))]
    p = [class2cnt[i] / len(classes) for i in range(5)]
    return np.array(vector), p


def classify(vec2cls, vecs, ps, classes):
    p_list = []
    for i in range(len(ps)):
        p_list.append(sum(vec2cls * vecs[i]) * ps[i])
    return p_list.index(max(p_list))


def word2vec(features, arr):
    r = [0] * 27
    base = 0
    for i, item in enumerate(arr):
        for j, jtem in enumerate(features[i]):
            if jtem == item:
                r[base + j] = 1
            else:
                r[base + j] = 0
        base += len(features[i])
    return r


def run():
    all, train_data, test_data, classes = load_data()
    train_mat = np.array([word2vec(features, item) for item in train_data])
    test_mat = np.array([word2vec(features, item) for item in test_data])
    vector, p = trainNB0(train_mat, classes)

    cnt = 0
    for i, item in enumerate(test_mat):
        c = classify(item, vector, p, classes)
        # print(c, classes[train_num+ i])
        if c == classes[train_num + i]:
            cnt += 1
    print(cnt / len(test_mat))


def t_word2vec():
    all, train_data, test_data, classes = load_data()
    train_mat = np.array([word2vec(features, item) for item in train_data])
    print(train_mat)


features = [
    ["usual", "pretentious", "great_pret"],
    ["proper", "less_proper", "improper", "critical", "very_crit"],
    ["complete", "completed", "incomplete", "foster"],
    ["1", "2", "3", "more"],
    ["convenient", "less_conv", "critical"],
    ["convenient", "inconv"],
    ["nonprob", "slightly_prob", "problematic"],
    ["recommended", "priority", "not_recom"]
]

if __name__ == '__main__':
    run()
    # t_word2vec()
