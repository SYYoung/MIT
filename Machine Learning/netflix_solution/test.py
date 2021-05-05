import numpy as np
# import em
import naive_em
import common

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")

testcase = 1
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
    K = 4
    n, d = X.shape
    seed = 0

# TODO: Your code here
