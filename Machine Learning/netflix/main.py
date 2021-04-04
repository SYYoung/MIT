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
print('K-means')
for i in np.arange(len(K_list)):
    for j in np.arange(len(seed_list)):
        ans1 = 'K-means: K = {}, seed = {}'.format(K_list[i], seed_list[j])
        print(ans1)
        mixture, post = common.init(X, K_list[i], seed_list[j])
        mixture, post, cost = kmeans.run(X, mixture, post)
        ans2 = 'Cost = {}'.format(cost)
        all_cost[i][j] = cost
        print(ans2)
    # common.plot(X, mixture, post, ans1)

print(all_cost)
print(np.min(all_cost, axis=1))


# mixture, post = common.init(X, K, seed)

## for EM algorithm
print('\n\nEM algorithm')
all_cost = np.zeros((len(K_list),len(seed_list)))
for i in np.arange(len(K_list)):
    for j in np.arange(len(seed_list)):
        ans1 = 'EM algorithm: K = {}, seed = {}'.format(K_list[i], seed_list[j])
        print(ans1)
        mixture, post = common.init(X, K_list[i], seed_list[j])
        mixture, post, likelihood = em.run(X, mixture, post)
        ans2 = 'Cost = {}'.format(likelihood)
        all_cost[i][j] = likelihood
        print(ans2)
    # common.plot(X, mixture, post, ans1)

print(all_cost)
print(np.min(all_cost, axis=1))

## calculate bic
best_likelihood = np.min(all_cost, axis=1)
bic_list = np.zeros(len(best_likelihood))
for i in np.arange(len(best_likelihood)):
    mixture, post = common.init(X, K_list[i], 0)
    bic_list[i] = common.bic(X, mixture, best_likelihood[i])
print(bic_list)
