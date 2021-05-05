import numpy as np
import em
import naive_em
import common

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")

testcase = 2
if (testcase == 1): # for naive_em
    X = np.loadtxt("toy_data.txt")
    K = 3
    seed = 0
    n, d = X.shape
    mixture, post = common.init(X, K, seed)
    mixture, post, ll = naive_em.run(X, mixture, post)
    result = "with naive_em, ll = {}".format(ll)
    print(result)


if (testcase == 2):
    X = np.loadtxt("netflix_incomplete.txt")
    # X = np.loadtxt("toy_data.txt")
    n, d = X.shape
    for K in [1, 12]:
        max_ll = None
        for seed in range(0, 5):
            ll = None
            mixture, post = common.init(X, K, seed)
            mixture, post, ll = em.run(X, mixture, post)
            if max_ll is None or ll > max_ll:
                max_ll = ll
                best_seed = seed
            ans = "K = {}, seed = {}, ll = {}".format(K, seed, ll)
            print(ans)

if (testcase == 3):
    X = np.loadtxt("netflix_incomplete.txt")
    mixture, post = common.init(X, 12, 0)
    mixture, post, ll = em.run(X, mixture, post)
    X_pred = em.fill_matrix(X, mixture)

    X_gold = np.loadtxt("netflix_complete.txt")
    rmse = common.rmse(X_gold, X_pred)
    print("RMSE between gold and incomplete = ", rmse)

# TODO: Your code here
# test BIC
    def run_with_bic():
        max_bic = None
        for K in range(1, 5):
            max_ll = None
            for seed in range(0, 5):
                mixture, post = common.init(X, K, seed)
                mixture, post, ll = naive_em.run(X, mixture, post)
                if max_ll is None or ll > max_ll:
                    max_ll = ll
                    best_seed = seed

            mixture, post = common.init(X, K, best_seed)
            mixture, post, ll = naive_em.run(X, mixture, post)
            bic = common.bic(X, mixture, ll)
            if max_bic is None or bic > max_bic:
                max_bic = bic
            title = "EM for K ={}, seed={}, ll={}, bic={}".format(K, best_seed, ll, bic)
            print(title)
            common.plot(X, mixture, post, title)
