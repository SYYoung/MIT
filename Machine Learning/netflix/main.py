import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
# 1. call common.init
K_list = [1,2,3,4]
seed_list = [0,1,2,3,4]
all_cost = np.zeros((len(K_list),len(seed_list)))
for i in np.arange(len(K_list)):
    for j in np.arange(len(seed_list)):
        ans1 = 'K = {}, seed = {}'.format(K_list[i], seed_list[j])
        print(ans1)
        mixture, post = common.init(X, K_list[i], seed_list[j])
        mixture, post, cost = kmeans.run(X, mixture, post)
        ans2 = 'Cost = {}'.format(cost)
        all_cost[i][j] = cost
        print(ans2)
        # common.plot(X, mixture, post, ans1)

print(all_cost)
print(np.min(all_cost, axis=1))
