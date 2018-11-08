import numpy as np


def str_feature_to_vector(arr: np.ndarray):
    row, column = np.shape(arr)
    uniq_col_val = [np.unique(arr[:, i]) for i in range(column - 1)]
    v2i_list = []
    for v in uniq_col_val:
        v2i_list.append({val: i for i, val in enumerate(v)})

    feature_cnt = sum([len(v) for v in uniq_col_val])
    result = np.zeros((row, feature_cnt + 1))
    for i in range(row):
        base = 0
        for j in range(column - 1):
            result[i][base + v2i_list[j][arr[i][j]]] = 1
            base += len(v2i_list[j])
        if arr[i][-1] == 'A':
            result[i][-1] = 1
        elif arr[i][-1] == 'S':
            result[i][-1] = 0
        else:
            result[i][-1] = -1
    return result


def str_feature_to_int_u(arr: np.ndarray, trans_list: list):
    row, column = np.shape(arr)
    # step 1:create str to int map for each column -> low:0 mid:1 high:2
    uniq_col_val = [np.unique(arr[:, i]) for i in trans_list]
    v2i_list = []
    for v in uniq_col_val:
        v2i_list.append({})
        for i in range(int(len(v) / 2)):
            int_val = len(v) - i * 2
            v2i_list[-1][v[i]] = -int_val
            v2i_list[-1][v[-i - 1]] = int_val
        if len(v) % 2 != 0:
            v2i_list[-1][v[int(len(v) / 2)]] = 0

    # step 2:give value for each item if item's column in list
    result = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            if j in trans_list:
                result[i][j] = v2i_list[trans_list.index(j)][arr[i][j]]
            else:
                try:
                    result[i][j] = float(arr[i][j])
                except:
                    result[i][j] = 0.0
    return result


def str_feature_to_int(arr: np.ndarray, trans_list: list):
    row, column = np.shape(arr)
    # step 1:create str to int map for each column -> low:0 mid:1 high:2
    uniq_col_val = [np.unique(arr[:, i]) for i in trans_list]
    v2i_list = []
    for v in uniq_col_val:
        v2i_list.append({v: i for i, v in enumerate(v)})

    # step 2:give value for each item if item's column in list
    result = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            if j in trans_list:
                result[i][j] = v2i_list[trans_list.index(j)][arr[i][j]]
            else:
                try:
                    result[i][j] = float(arr[i][j])
                except:
                    result[i][j] = 0.0
    return result


def split_by_label_map(arr: np.ndarray):
    uniq_label = np.unique(arr[:, -1])
    label2array = {}
    for label in uniq_label:
        label2array[label] = arr[np.where(arr[:, -1] == label)]
    return label2array


def split_by_label(arr: np.ndarray, index:int):
    uniq_label = np.unique(arr[:, index])
    return [arr[np.where(arr[:, index] == label)] for label in uniq_label]


if __name__ == '__main__':
    # arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 3]).reshape((3, 3))
    # print(arr)
    # print(split_by_label(arr))
    #
    arr = np.loadtxt('post-operative.data', delimiter=',', dtype=np.unicode)
    # result = str_feature_to_int(arr, [0, 1, 2, 3, 4, 5, 6, 8])
    # print("ok")
    result = str_feature_to_vector(arr)
    print("o")
