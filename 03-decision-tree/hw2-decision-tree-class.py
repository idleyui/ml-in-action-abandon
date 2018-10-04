import numpy as np

from collections import Counter
from math import log
import operator
import pickle


def calc_shannon_ent(classes):
    classes_counter = Counter(classes)
    shannon_ent = 0.0
    for key, value in classes_counter.items():
        prob = float(value) / len(classes)
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# for test only
def create_dataset():
    data = [[1, 1, 'yes'], [0, 1, 'yes'], [1, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], ]
    labels = ['no surfacing', 'flippers']
    classes = [item[-1] for item in data]
    return data, labels, classes


def split_dataset(data, axis, value):
    return [item[:axis] + item[axis + 1:] for item in data if item[axis] == value]


def dataset_filter(data, filter):
    return [item for item in data if filter(item)]


def load_data():
    data = np.loadtxt('page-blocks.data', delimiter=',')
    labels = ["height", "lenght", "area", "eccen", "p_black",
              "p_and", "mean_tr", "blackpix", "blackand", "wb_trans"]
    classes = [item[-1] for item in data]
    return data, labels, classes


def choose_feature_to_split(data, labels, classes):
    base_ent = calc_shannon_ent(classes)
    base_info_gain = 0.0
    best_feature = -1
    for i in range(len(labels)):
        uni_features = set([item[i] for item in data])
        new_ent = 0.0
        for val in uni_features:
            sub = split_dataset(data, i, val)
            prob = len(sub) / float(len(data))
            new_ent += prob * calc_shannon_ent(sub)
        info_gain = base_ent - new_ent
        if info_gain > base_info_gain:
            base_info_gain = info_gain
            best_feature = i
    return best_feature


def reg_feature_to_split(data, labels, classes):
    base_env = calc_shannon_ent(classes)
    base_info_gain = 0.0
    best_feature = -1
    for i in range(len(labels)):
        # get unique & sorted features
        features = sorted(list(set([item[i] for item in data])))
        new_ent = 0.0
        for i in range(len(features) - 1):
            boundary = (features[i] + features[i + 1]) / 2
            sub_g = dataset_filter(data, lambda item: item[i] > boundary)
            sub_l = dataset_filter(data, lambda item: item[i] <= boundary)
            prob_g = len(sub_g) / float(len(data))
            prob_l = len(sub_l) / float(len(data))
            new_ent = prob_g * calc_shannon_ent(sub_g) + prob_l * calc_shannon_ent(sub_l)
        info_gain = base_env - new_ent
        if info_gain > base_info_gain:
            base_info_gain = info_gain
            best_feature = i
    return best_feature


def tree(data, labels, classes, features):
    if classes.count(classes[0]) == len(classes):
        return classes[0]
    if len(data[0]) == 1:
        return Counter(classes).most_common(1)[0][0]

    best_feature = choose_feature_to_split(data, labels, classes)
    label = labels[best_feature]

    features.append(label)
    my_tree = {label: {}}
    del (labels[best_feature])
    uniq_vals = set([item[best_feature] for item in data])
    for v in uniq_vals:
        my_tree[label][v] = tree(data, labels, classes, features)
    return my_tree


def classify(tree, features, test):
    label = ''
    first_str = next(iter(tree))
    second_dic = tree[first_str]
    feature_i = features.index(first_str)
    for key, value in second_dic.items():
        if test[feature_i] == key:
            if type(value).__name__ == 'dict':
                label = classify(value, features, test)
            else:
                label = value
    return label


if __name__ == '__main__':
    print(calc_shannon_ent([1, 1, 1, 2, 2, 2]))
    print(calc_shannon_ent([1, 1]))
    print(calc_shannon_ent([1, 2, 2, 2]))
    # data, labels, classes = create_dataset()
    # features = []
    # t = tree(data, labels, classes, features)
    #
    # test = []
    # result = classify(tree, features, test)
    #
    # print(split_dataset(data, 1, 1))
    # print(data)
