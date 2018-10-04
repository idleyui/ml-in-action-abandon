import numpy as np

from collections import Counter
from math import log
from timeit import default_timer as timer


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


# for test only
def create_dataset():
    data = [[1, 1, 'yes'], [0, 1, 'yes'], [1, 0, 'no'], [0, 0, 'no'], [0, 0, 'no'], ]
    labels = ['no surfacing', 'flippers']
    classes = get_class(data)
    return data, labels, classes


def dataset_filter(data, filter):
    return [item for item in data if filter(item)]


def load_data():
    data = np.loadtxt('page-blocks.data')
    labels = ["height", "lenght", "area", "eccen", "p_black",
              "p_and", "mean_tr", "blackpix", "blackand", "wb_trans"]
    return data, labels


def feature_to_split(data, labels, classes):
    base_env = calc_shannon_ent(data)
    base_info_gain = 0.0
    best_feature = -1
    bound = 0.0
    for index in range(len(labels)):
        features = sorted([item[index] for item in data])
        for i in range(len(features) - 1):
            if classes[i + 1] == classes[i] or features[i + 1] == features[i]:
                continue
            # print(index, i)
            boundary = (features[i] + features[i + 1]) / 2
            sub_g = dataset_filter(data, lambda item: item[index] > boundary)
            sub_l = dataset_filter(data, lambda item: item[index] <= boundary)
            prob_g = len(sub_g) / float(len(data))
            prob_l = len(sub_l) / float(len(data))
            new_ent = prob_g * calc_shannon_ent(sub_g) + prob_l * calc_shannon_ent(sub_l)
            info_gain = base_env - new_ent
            if info_gain > base_info_gain:
                base_info_gain = info_gain
                best_feature = index
                bound = boundary
    return best_feature, base_info_gain, bound


def tree(data, labels, features):
    classes = get_class(data)
    if classes.count(classes[0]) == len(classes):
        return classes[0]
    # if len(data) < 5:
    #     return Counter(classes).most_common(1)[0][0]

    best_feature, info_gain, boundary = feature_to_split(data, labels, classes)
    if info_gain < 0.0001:
        return Counter(classes).most_common(1)[0][0]

    label = labels[best_feature]

    features.append(label)
    print(len(data))
    my_tree = {'label': label, 'val': boundary,
               'left': tree(dataset_filter(data, lambda item: item[best_feature] <= boundary),
                            labels, features),
               'right': tree(dataset_filter(data, lambda item: item[best_feature] > boundary),
                             labels, features)
               }

    return my_tree


def classify(tree, labels, test):
    if isinstance(tree, float):
        return test[-1] == tree, tree
    if tree['val'] >= test[labels.index(tree['label'])]:
        return classify(tree['left'], labels, test)
    else:
        return classify(tree['right'], labels, test)


if __name__ == '__main__':
    data, labels = load_data()
    t = np.vsplit(data, np.array([1]))
    train = t[0]
    test = t[1]

    start = timer()

    features = []
    t = tree(train, labels, features)
    # print(t)
    end = timer()
    print('use %ds to create the tree' % (end - start))
    cnt = [0, 0, 0, 0, 0]
    right = [0, 0, 0, 0, 0]
    for item in test:
        result, value = classify(t, labels, item)
        cnt[int(item[-1] - 1)] += 1
        if result:
            right[int(item[-1] - 1)] += 1
        else:
            print(item[-1], value)

    print('Use %d test case, %d is right, accuracy is %.2f' % (sum(cnt), sum(right), sum(right) / sum(cnt)))
    for i in range(5):
        if cnt[i] != 0:
            print('Use %d test case for %d, %d is right, accuracy is %.2f' %
                  (cnt[i], i + 1, right[i], right[i] / cnt[i]))
