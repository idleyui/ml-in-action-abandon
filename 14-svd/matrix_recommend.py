import numpy as np
from numpy import linalg as la
from scipy.sparse.linalg import svds
import timeit
import pickle
import argparse

parser = argparse.ArgumentParser(description="Web Page Recommendation")
parser.add_argument('--run', type=str, default='build')  # build | test
parser.add_argument('--est_method', type=str, default='all')  # std | svd | all
parser.add_argument('--N', type=int, default=10)  # top N for test < 10


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
    if denom == 0:
        return 0
    return 0.5 + 0.5 * (num / denom)


class Recommend:

    def __init__(self, X, est_func=None, est_strategy='avg', similar_func=cosSim):
        self.X = X
        self.U, self.Sigma, self.VT = svds(self.X)
        self.Sig4 = np.mat(np.eye(4) * self.Sigma[:4])
        self.xformedItems = self.X.T * self.U[:, :4] * self.Sig4.I
        self.similar_func = similar_func
        self.est_func = self.svd_est if est_func is not None else self.std_est
        self.n = np.shape(X)[1]
        self.similar_matrix = np.zeros((self.n, self.n))
        self.build_similar_matrix()
        self.est_strategy = est_strategy

    def build_similar_matrix(self):
        for i in range(self.n):
            for j in range(i, self.n):
                # try:
                score = self.est_func(i, j)
                self.similar_matrix[i][j] = score
                self.similar_matrix[j][i] = score
            # except:
            #     print(i, j)

    def std_est(self, item, rated_item):
        # overLap = np.nonzero(np.logical_and(self.X[:, item].A > 0, self.X[:, rated_item].A > 0))[0]
        overLap = np.nonzero(np.logical_or(self.X[:, item].A > 0, self.X[:, rated_item].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = self.similar_func(self.X[overLap, item], self.X[overLap, rated_item])
            # similarity = self.similar_func(self.X[:, item], self.X[:, rated_item])
        return similarity

    def svd_est(self, item, rated_item):
        return self.similar_func(self.xformedItems[item, :].T, self.xformedItems[rated_item, :].T)
        # return self.similar_func(self.xformedItems[item, :].T, self.xformedItems[rated_item, :].T)

    def estimate(self, item, user):
        sim_cnt = 0
        ratSimTotal = 0.0
        for j in np.where(self.X[user] > 0)[1]:
            similarity = self.similar_matrix[item, j]
            # print('the %d and %d similarity is: %f' % (item, j, similarity))
            # sim_cnt += similarity
            # average strategy
            sim_cnt += 1
            ratSimTotal += similarity * self.X[user, j]
        if sim_cnt == 0:
            return 0
        else:
            return ratSimTotal / sim_cnt

    def max_estimate(self, item, user):
        max = 0
        for j in np.where(self.X[user] > 0)[1]:
            similarity = self.similar_matrix[item, j]
            # print('the %d and %d similarity is: %f' % (item, j, similarity))
            # sim_cnt += similarity
            # average strategy
            if similarity > max:
                max = similarity
        return max

    def recommend(self, user, N=10):
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
        if user % 500 == 0:
            print('recommend for user ', user)
        unrated_items = np.where(self.X[user] == 0)[1]  # find unrated items
        if len(unrated_items) == 0:
            print('user %d has rated everything, no new recommend' % user)
            return []

        item_scores = []
        for item in unrated_items:
            if self.est_strategy == 'avg':
                score = self.estimate(item, user)
            else:
                score = self.max_estimate(item, user)
            # score = self.est_func(user, item)
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
    label = []
    for user in userlist:
        user2attr = np.zeros(len(attrlist))
        for attr in user2visited[user]:
            user2attr[attr2index[attr]] = 1
        mx.append(user2attr)
        label.append(user)
    return label, np.array(mx)


def est_recommends(label2index, recommends, test_label, test):
    total_visited = 0
    total_recommend = 0
    for i in range(len(test)):
        j = label2index[test_label[i]]
        visited = 0
        for item in recommends[j]:
            if test[i][item[0]] > 0:
                visited += 1
        total_visited += visited
        total_recommend += len(recommends[i])
        print('recommend %d for user %d, visited %d' % (len(recommends[i]), test_label[i], visited))
    return total_recommend, total_visited


def build_recommend_matrix(func, file, str):
    # data = load_data('../Datasets/anonymous-msweb/anonymous-msweb.data')
    label, data = load_data('data/anonymous-msweb.data')

    t1 = timeit.default_timer()
    r = Recommend(np.mat(data), func, str)
    t2 = timeit.default_timer()
    print('use %d seconds to build similar matrix' % (t2 - t1))

    rec = [r.recommend(i) for i in range(len(data))]
    t3 = timeit.default_timer()
    print('use %d seconds build recommends matrix by std mathod' % (t3 - t2))
    print('use %d seconds to build recommend matrix' % (t3 - t1))

    with open(file, 'wb') as f:
        pickle.dump(rec, f)


def test_recommend(file, N=10):
    """Recommendation test

    :param file: file to load recommend matrix
    :param N: use Top N to test
    :return:
    """
    with open(file, 'rb') as f:
        recommends = pickle.load(f)

    test_label, test = load_data('data/anonymous-msweb.test')
    label, _ = load_data('data/anonymous-msweb.data')
    label2index = {label: i for i, label in enumerate(label)}

    total_visited = 0
    total_recommend = 0
    visited_user = 0
    for i in range(len(test)):
        j = label2index[test_label[i]]
        visited = 0
        for r in range(N):
            item = recommends[j][r]
            # for item in recommends[j]:
            if test[i][item[0]] > 0:
                visited += 1
        if visited > 0:
            visited_user += 1
        total_visited += visited
        total_recommend += len(recommends[i])
        # print('recommend %d for user %d, visited %d' % (len(recommends[i]), test_label[i], visited))

    # total_recommend, total_visited = est_recommends(label2index, , label, test)
    print('recommend for %d users, %d users visited recommend page, rate: %f' % (
        len(test), visited_user, visited_user / len(test)))
    print('recommend %d for users, visited %d, visited rate: %f' % (
        total_recommend, total_visited, total_visited / total_recommend))


def main():
    args = parser.parse_args()

    if args.run == 'build':
        if args.est_method == 'all':
            build_recommend_matrix(None, 'data/std_avg_mx', 'avg')
            build_recommend_matrix('svd', 'data/svd_avg_mx', 'avg')
        elif args.est_mehtod == 'std':
            build_recommend_matrix(None, 'data/std_avg_mx', 'avg')
        elif args.est_mehtod == 'svd':
            build_recommend_matrix('svd', 'data/svd_avg_mx', 'avg')
    else:
        if args.N > 10 or args.N < 1:
            print('N is invalid')
        else:
            test_recommend('data/std_avg_mx', args.N)
            test_recommend('data/svd_avg_mx', args.N)


if __name__ == '__main__':
    main()
