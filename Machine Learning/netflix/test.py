import numpy as np
import em
import common

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

testcase = 2
if (testcase == 1):
    X = np.loadtxt("test_incomplete.txt")
    K = 4
elif (testcase == 2):
    X = np.loadtxt("netflix_incomplete.txt")
    K = 1
    seed = 0
else:
    X = np.loadtxt("netflix_complete.txt")
    K = 3
    seed = 0

# X_gold = np.loadtxt("test_complete.txt")

# X = np.loadtxt("toy_data.txt")
# K = 6
seed = 0

# K = 6

n, d = X.shape

# TODO: Your code here
# for testing
# mixture, post = common.test_setup(X, K, seed)
iter = 1
loglikelihood_list = np.zeros(100)
mixture, post = common.init(X, K, seed)


for i in np.arange(iter):
    mixture, post, likelihood = em.run(X, mixture, post)
    ans2 = 'New log likelihood = {}'.format(likelihood)
    loglikelihood_list[i] = likelihood
    print('mu = \n', mixture.mu)
    print('variance = \n', mixture.var)
    print('p = \n', mixture.p)

print(loglikelihood_list)
plt.plot(loglikelihood_list)

# fill in the missing entries
X_predicted = em.fill_matrix(X, mixture)

# for EM algorithm
if (testcase == 2):
    X = np.loadtxt("netflix_incomplete.txt")
    K_list = [1,12]
    seed_list = [0,1,2,3,4]
    all_cost = np.zeros((len(K_list),len(seed_list)))
    print('\n\nEM algorithm')
    for i in np.arange(len(K_list)):
        for j in np.arange(len(seed_list)):
            ans1 = 'EM algorithm: K = {}, seed = {}'.format(K_list[i], seed_list[j])
            print(ans1)
            mixture, post = common.init(X, K_list[i], seed_list[j])
            mixture, post, likelihood = em.run(X, mixture, post)
            ans2 = 'LogLikelihood = {}'.format(likelihood)
            all_cost[i][j] = likelihood
            print(ans2)
        # common.plot(X, mixture, post, ans1)

    print(all_cost)
    print(np.min(all_cost, axis=1))
