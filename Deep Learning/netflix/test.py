import numpy as np
import em
import common

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")
import kmeans

#K = 4
#n, d = X.shape
import naive_em

seed = 0

# TODO: Your code here
# test for k-means
def test_k_means():
    print("2 K-means.")
    X = np.loadtxt("toy_data.txt")
    for K in [1,2,3,4]:
        min_cost = None
        best_seed = None
        for seeds in range(5):
            mixture, post = common.init(X, K, seeds)
            mixture, post, cost = kmeans.run(X, mixture, post)
            if min_cost is None or cost < min_cost:
                best_seed = seed
                min_cost = cost
        mixture, post = common.init(X, K, best_seed)
        mixture, post, cost = kmeans.run(X, mixture, post)
        title = "K-means for K={}, seed={}, cost={}".format(K, best_seed, cost)
        print(title)
        common.plot(X, mixture, post, title)

# test_k_means()

def test_naive_em():
    print("testing naive_em")
    X = np.loadtxt("toy_data.txt")
    K = 3
    seed = 0

    mixture, post = common.init(X, K, seed)
    mixture, post, ll = naive_em.run(X, mixture, post)

test_naive_em()