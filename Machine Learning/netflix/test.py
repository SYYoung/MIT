import numpy as np
import em
import common

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")

X = np.loadtxt("toy_data.txt")
K = 3
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

print(loglikelihood_list)
plt.plot(loglikelihood_list)
