import numpy as np

if __name__ == '__main__':
    state = np.array([0.65, 0.15, 0.12, 0.28, 0.67, 0.36, 0.07, 0.18, 0.52]) \
        .reshape((3, 3)).T
    # 随机生成10次初始分布，对每个初始分布迭代10次，输出迭代结果
    for i in range(10):
        p = np.random.dirichlet(np.ones(3), 1)[0]
        print('test %d. start: %s' % (i, p), end='\t')
        for j in range(10):
            p = np.dot(p, state)
        print('end:', p)
