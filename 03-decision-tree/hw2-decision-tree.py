import numpy as np

from collections import Counter
from math import log
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def get_class(data):
    return [item[-1] for item in data]


def calc_shannon_ent(data):
    classes = get_class(data)
    classes_counter = Counter(classes)
    shannon_ent = 0.0
    for key, value in classes_counter.items():
        prob = float(value) / len(classes)
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# return data and labels
def load_data():
    return np.loadtxt('data/page-blocks.data'), ["height", "lenght", "area", "eccen", "p_black",
                                                 "p_and", "mean_tr", "blackpix", "blackand", "wb_trans"]


# choose the feature with best information gain
# two step: find the point with the max info main in one feature -> find the max in all features
def feature_to_split(data, labels):
    base_env = calc_shannon_ent(data)
    base_info_gain = 0.0
    best_feature = -1
    bound = 0.0

    max_len = 0
    min_shannon = 999999
    for index in range(len(labels)):
        d = data[:, [index, -1]]
        features = d[np.argsort(d[:, 0])[::-1]]
        for i in range(len(features) - 1):
            if features[i + 1][1] == features[i][1] or features[i + 1][0] == features[i][0]:
                continue
            # print(index, i)
            boundary = (features[i][1] + features[i + 1][1]) / 2
            sub_g = data[data[:, index] > boundary]
            sub_l = data[data[:, index] <= boundary]
            prob_g = len(sub_g) / float(len(data))
            prob_l = len(sub_l) / float(len(data))
            new_ent = prob_g * calc_shannon_ent(sub_g) + prob_l * calc_shannon_ent(sub_l)
            info_gain = base_env - new_ent
            if info_gain > base_info_gain:
                base_info_gain = info_gain
                best_feature = index
                bound = boundary
            # shan_g = calc_shannon_ent(sub_g)
            # shan_l = calc_shannon_ent(sub_l)
            # if len(sub_g) == 0 or len(sub_l) == 0:
            #     continue
            # if shan_g < shan_l:
            #     sub = shan_g
            #     l = len(sub_g)
            # elif shan_g > shan_l:
            #     sub = shan_l
            #     l = len(sub_l)
            # elif len(sub_g) > len(sub_l):
            #     sub = shan_g
            #     l = len(sub_g)
            # else:
            #     sub = shan_l
            #     l = len(sub_l)
            #
            # if sub < min_shannon or (sub == min_shannon and l > max_len):
            #     min_shannon = sub
            #     max_len = l
            #     best_feature = index
            #     bound = boundary
            #     base_info_gain = shan_l + shan_g

    return best_feature, base_info_gain, bound


# find feature with max info gain -> 1. if info gain too small, return most common class, 2. split and recur
def tree(data, labels, info):
    c = Counter(get_class(data)).get(3)
    if c is not None:
        print(len(data), c, info)
    best_feature, info_gain, boundary = feature_to_split(data, labels)
    # almost to one side(all the same & only one & etc)
    if info_gain < 0.0001:
        return Counter(get_class(data)).most_common(1)[0][0]

    return {'label': labels[best_feature], 'val': boundary,
            'data': str(Counter(get_class(data))),
            'left': tree(data[data[:, best_feature] <= boundary], labels, info + 1),
            'right': tree(data[data[:, best_feature] > boundary], labels, info + 1)
            }


def classify(tree, labels, test):
    if isinstance(tree, float):
        return test[-1] == tree, tree
    if tree['val'] >= test[labels.index(tree['label'])]:
        return classify(tree['left'], labels, test)
    else:
        return classify(tree['right'], labels, test)


def decision_tree():
    # load data
    data, labels = load_data()
    c = [0, 0, 0, 0, 0]
    for key, value in Counter(get_class(data)).items():
        c[int(key - 1)] += int(value / 5)

    c[0] = 4000
    train = np.zeros((0, 11))
    test = np.zeros((0, 11))

    for item in data:
        if c[int(item[-1] - 1)] > 0:
            # test = np.insert(test, 0, values=item, axis=0)
            c[int(item[-1] - 1)] -= 1
        else:
            train = np.insert(train, 0, values=item, axis=0)
            test = np.insert(test, 0, values=item, axis=0)

    # t = np.vsplit(data, np.array([5000]))
    # train = t[0]
    # test = t[1]

    # create tree
    start = timer()
    t = tree(train, labels, 0)
    end = timer()
    print('use %ds to create the tree' % (end - start))
    f = open('out/tree.json', 'w')
    f.write(str(t))

    # test
    cnt = [0, 0, 0, 0, 0]
    right = [0, 0, 0, 0, 0]
    for item in test:
        value = int(item[-1] - 1)
        cnt[value] += 1
        success, label = classify(t, labels, item)
        if success:
            right[value] += 1
        # else:
        # print(value, label)

    # print result
    print('Use %d test case, %d is right, accuracy is %.2f' % (sum(cnt), sum(right), sum(right) / sum(cnt)))
    for i in range(5):
        if cnt[i] != 0:
            print('Use %d test case for %d, %d is right, accuracy is %.2f' %
                  (cnt[i], i + 1, right[i], right[i] / cnt[i]))


if __name__ == '__main__':
    decision_tree()
