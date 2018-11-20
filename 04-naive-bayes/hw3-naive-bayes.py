from collections import Counter

import numpy as np


def load_data(train_num=10000):
    d = np.loadtxt('data/nursery.data', delimiter=',', dtype=np.unicode)
    np.random.shuffle(d)
    train, test = np.vsplit(np.delete(d, -1, axis=1), np.array([train_num]))
    mp = {item: i for i, item in enumerate(list(np.unique(d[:, -1])))}
    classes = np.array([mp[item] for item in d[:, -1]])
    return np.delete(d, -1, axis=1), train, test, classes


def trainNB0(data, classes):
    # get prior probability
    class2cnt = Counter(classes)
    p = np.array([class2cnt[i] / len(classes) for i in range(len(class2cnt))])

    # get conditional probability
    c_len = len(class2cnt)
    word_in_class_vector = np.zeros((c_len, len(data[0])))
    cnt_word_in_class = np.zeros(c_len)
    for i, doc in enumerate(data):
        word_in_class_vector[classes[i]] += doc
        cnt_word_in_class[classes[i]] += sum(doc)
    vector = [word_in_class_vector[i] / cnt_word_in_class[i] for i in range(c_len)]
    return np.array(vector), p


def classify(test_item, vector, p):
    p_list = [sum(test_item * vector[i]) * p[i] for i in range(len(p))]
    return p_list.index(max(p_list))


def word2vec(dic, arr):
    r = [0] * len(dic)
    for item in arr:
        r[dic[item]] = 1
    return r


def dict_bayes(train_num=10000):
    all, train_data, test_data, classes = load_data(train_num)
    dic = {item: i for i, item in enumerate(np.unique(all))}
    train_mat = np.array([word2vec(dic, item) for item in train_data])
    test_mat = np.array([word2vec(dic, item) for item in test_data])
    vector, p = trainNB0(train_mat, classes)

    cnt = 0
    for i, item in enumerate(test_mat):
        c = classify(item, vector, p)
        # print(c, classes[train_num + i])
        if c == classes[train_num + i]:
            cnt += 1
    print("%.2f" % (cnt / len(test_mat)))


def feature2vec(features, arr):
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


def feature_bayes(train_num=10000):
    all, train_data, test_data, classes = load_data(train_num)
    train_mat = np.array([feature2vec(features, item) for item in train_data])
    test_mat = np.array([feature2vec(features, item) for item in test_data])
    vector, p = trainNB0(train_mat, classes)

    cnt = 0
    for i, item in enumerate(test_mat):
        c = classify(item, vector, p)
        # print(c, classes[train_num+ i])
        if c == classes[train_num + i]:
            cnt += 1
    print("%.2f" % (cnt / len(test_mat)))


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
    for i in range(9000, 13000, 1000):
        print('use {:2.2%} item for train'.format(i / 12960))
        print('use vocab list to train:', end='\t')
        dict_bayes(i)
        print('use feature list to train:', end='\t')
        feature_bayes(i)
