import numpy as np
import em
import common

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")

# X = np.loadtxt("toy_data.txt")
# K = 3
seed = 0

# this is the test case
X = np.array( [[0.85794562, 0.84725174],  [0.6235637,  0.38438171],
                [0.29753461, 0.05671298], [0.27265629, 0.47766512],
                [0.81216873, 0.47997717], [0.3927848,  0.83607876],
                [0.33739616, 0.64817187], [0.36824154, 0.95715516],
                [0.14035078, 0.87008726], [0.47360805, 0.80091075],
                [0.52047748, 0.67887953], [0.72063265, 0.58201979],
                [0.53737323, 0.75861562], [0.10590761, 0.47360042],
                [0.18633234, 0.73691818]])
K = 6

n, d = X.shape

# TODO: Your code here
iter = 3
loglikelihood_list = np.zeros(100)
mixture, post = common.test_setup(X, K, seed)

for i in np.arange(iter):
    mixture, post, likelihood = em.run(X, mixture, post)
    ans2 = 'New log likelihood = {}'.format(likelihood)
    loglikelihood_list[i] = likelihood
    print(ans2)

print(loglikelihood_list)
plt.plot(loglikelihood_list)
