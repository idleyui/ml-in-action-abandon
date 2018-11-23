import numpy as np
from numpy import linalg as la
from scipy.sparse.linalg import svds
import timeit


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# pearson correlation
def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


# cosine similarity
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


class Recommend:

    def __init__(self, X, similar_func=cosSim, est_func=None):
        self.X = X
        self.U, self.Sigma, self.VT = svds(self.X)
        self.Sig4 = np.mat(np.eye(4) * self.Sigma[:4])
        self.xformedItems = self.X.T * self.U[:, :4] * self.Sig4.I
        self.similar_func = similar_func
        self.est_func = est_func if est_func is not None else self.std_est
        n = np.shape(X)[1]
        self.similar_matrix = np.zeros((n, n))

    def std_est(self, item, rated_item):
        overLap = np.nonzero(np.logical_and(self.X[:, item].A > 0, self.X[:, rated_item].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            # similarity = self.similar_func(self.X[overLap, item], self.X[overLap, rated_item])
            similarity = cosSim(self.X[overLap, item], self.X[overLap, rated_item])
        return similarity

    def svd_est(self, item, rated_item):
        return self.similar_func(self.xformedItems[item, :].T, self.xformedItems[rated_item, :].T)

    def estimate(self, item, user):
        sim_cnt = 0
        ratSimTotal = 0.0
        for j in np.where(self.X[user] > 0)[1]:
            similarity = self.est_func(item, j)
            print('the %d and %d similarity is: %f' % (item, j, similarity))
            sim_cnt += similarity
            ratSimTotal += similarity * data[user, j]
        if sim_cnt == 0:
            return 0
        else:
            return ratSimTotal / sim_cnt

    def recommend(self, user, N=3):
        """Recommend top N similar item for user

        steps:
            1. find all unrated items
            2. calculate similarity of unrated items with user
            3. sort unrated items by similarity and return top N

        :param data: data matrix
        :param user: user id
        :param N: top N
        :param similar_func: similar method
        :param est_func: estimate method
        :return: top N items index
        """
        unrated_items = np.where(self.X[user] == 0)[1]  # find unrated items
        if len(unrated_items) == 0:
            print('user %d has rated everything, no new recommend' % user)
            return []

        item_scores = []
        for item in unrated_items:
            score = self.est_func(user, item)
            item_scores.append((item, score))
        return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]


def load_data(path):
    with open(path) as f:
        lines = f.readlines()

    user2visited = {}
    attr_map = {}
    ignore_set = {'I', 'T', 'N'}
    uid = -1
    for line in lines:
        if line[0] in ignore_set:
            continue
        if line[0] == 'A':
            attr = line.split(',')
            attr_map[int(attr[1])] = attr
        elif line[0] == 'C':
            uid = int(line.split(',')[2])
        elif line[0] == 'V':
            if uid in user2visited.keys():
                user2visited[uid].append(int(line.split(',')[1]))
            else:
                user2visited[uid] = [int(line.split(',')[1])]
    userlist = sorted(user2visited.keys())
    attrlist = sorted(attr_map.keys())
    attr2index = {val: index for index, val in enumerate(attrlist)}
    mx = []
    for user in userlist:
        user2attr = np.zeros(len(attrlist))
        for attr in user2visited[user]:
            user2attr[attr2index[attr]] = 1
        mx.append(user2attr)
    return np.array(mx)


def est_recommends(recommends, test):
    total_visited = 0
    total_recommend = 0
    for user in test:
        visited = 0
        for item in recommends:
            if test[user][item[0]] > 0:
                visited += 1
        total_visited += visited
        total_recommend += len(recommends)
        print('recommend %d for user %d, visited %d' % (len(recommends), user, visited))
    print('recommend %d for users, visited %d' % (total_recommend, total_visited))
    print('visited rate: %f' % total_visited / total_recommend)


if __name__ == '__main__':
    data = load_data('../Datasets/anonymous-msweb/anonymous-msweb.data')
    test = load_data('../Datasets/anonymous-msweb/anonymous-msweb.test')

    r = Recommend(np.mat(data), Recommend.std_est)
    std_recommends = [r.recommend(i) for i in range(len(data))]
    r.est_func = Recommend.svd_est
    svd_recommends = [r.recommend(i) for i in range(len(data))]
