"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal as norm


def naive_estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.array, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # 1. build nn: normal distribution n*K
    mu = mixture.mu
    p = mixture.p
    var = mixture.var
    num = X.shape[0]
    K = len(p)

    # 1. build nn
    norm_table = np.array([norm.pdf(X[0], mean=mu[j], cov=(var[j])) for j in np.arange(K)])
    for i in np.arange(1, num):
        nn = np.array([norm.pdf(X[i], mean=mu[j], cov=(var[j])) for j in np.arange(K)])
        norm_table = np.append(norm_table, nn)
    norm_table = norm_table.reshape((num, K))
    # 2. calculate px
    px = np.matmul(norm_table, np.transpose(p))
    # 3. calculate pj_x: the soft count
    pj_x = norm_table * p
    pj_x = pj_x/px.reshape((num, 1))
    # 4. calculate the log likelihood
    log_px = np.log(px)
    likelihood = np.sum(log_px)

    return pj_x, likelihood


def naive_mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    num, d = X.shape
    K = post.shape[1]
    nj = np.sum(post, axis=0)
    pj = nj/num
    u = np.matmul(np.transpose(post), X)
    mu = u/nj.reshape(len(nj), 1)

    post_sum = np.sum(post, axis=0)
    var = np.zeros(K)

    for j in np.arange(K):
        diff = X - mu[j]
        diff = np.linalg.norm(diff, axis=1) ** 2
        v = np.sum(post[:,j] * diff)
        # v = np.transpose(post[:,j]) * diff
        var[j] = v/(d * nj[j])

    new_mixture = GaussianMixture(mu, var, pj)
    return new_mixture

def naive_run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    l_old = 0
    not_converge = True
    while (not_converge):
        post, loglikehood = naive_estep(X, mixture)
        mixture = naive_mstep(X, post, mixture)
        ans = 'logLikelihood = {}'.format(loglikehood)
        # print(ans)
        not_converge = np.abs(loglikehood - l_old) >= (10**-6 * np.abs(loglikehood))
        l_old = loglikehood
    return mixture, post, loglikehood

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.array, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # 1. build nn: normal distribution n*K
    mu = mixture.mu
    p = mixture.p
    var = mixture.var
    num, d  = X.shape
    K = len(p)

    # 0. build the map which only includes Cu
    cu = []
    for n in np.arange(num):
        mask_flag = [i for i in np.arange(d) if X[n,i] != 0]
        cu.append(mask_flag)
    logp_j = np.transpose(np.log(p + 1e-16))

    # 1. build nn
    lj_x = []
    loglikelihood = 0

    for i in np.arange(num):
        # for numerical stability, use logsumexp and take out the average
        if (len(cu[i]) == 0):
            fu_j = logp_j
            lj_u = fu_j
        else:
            nn = np.array([norm.pdf(X[i,cu[i]], mean=mu[j,cu[i]], cov=(var[j])) for j in np.arange(K)])
            # for numerical stability, if nn is too small, set fu_j and lj_u to be logp_j
            if (np.sum(nn) < 1e-6):
                fu_j = logp_j
                lj_u = fu_j
            else:
                fu_j = logp_j + np.log(nn)
                x_max = max(fu_j)
                lj_u = fu_j - (x_max + logsumexp(fu_j - x_max))
        lj_x.append(lj_u)
        loglikelihood = loglikelihood + np.dot((fu_j - lj_u), np.exp(lj_u))
    lj_x = np.array(lj_x).reshape((num, K))

    pj_x = np.exp(lj_x)

    return pj_x, loglikelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    num, d = X.shape
    K = post.shape[1]
    nj = np.sum(post, axis=0)
    pj = nj/num

    # 1. build up delta array
    cu = []
    delta_list = []
    cu_mag = []
    for n in np.arange(num):
        mask_flag = [i for i in np.arange(d) if X[n, i] != 0]
        delta_flag = [1 if X[n, i] != 0 else 0 for i in np.arange(d) ]
        cu.append(mask_flag)
        delta_list.append((delta_flag))
        cu_mag.append(len(mask_flag))
    delta = np.array(delta_list).reshape((num, d))

    # 2. calculate mu
    mu = np.zeros((K, d))
    for i in np.arange(K):
        temp = np.reshape(post[:,i], (num, 1))
        temp = temp * delta * X
        mu[i] = np.sum(temp, axis=0) + 1e-16

    denom = np.matmul(np.transpose(post), delta) + 1e-16
    mu = mu / denom
    # 3. replace the old mu if it passes the criteria
    mu_old = mixture.mu
    for j in np.arange(K):
        for feature in np.arange(d):
            if (denom[j,feature] < 1):
                mu[j,feature] = mu_old[j,feature]

    post_sum = np.sum(post, axis=0)
    v = np.zeros((num))
    var = np.zeros(K)

    # 4. calculate variance
    norm_val = np.zeros(num)
    for j in np.arange(K):
        for i in np.arange(num):
            diff = X[i,cu[i]] - mu[j,cu[i]]
            diff = np.linalg.norm(diff) ** 2
            v[i] = diff
        numerator = np.dot(post[:,j], v) + 1e-16
        denom = np.dot(post[:,j], np.array(cu_mag)) + 1e-16
        var[j] = numerator / denom
        if (var[j] < min_variance):
            var[j] = min_variance

    new_mixture = GaussianMixture(mu, var, pj)
    return new_mixture

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    l_old = 0
    not_converge = True
    while (not_converge):
        post, loglikehood = estep(X, mixture)
        mixture = mstep(X, post, mixture)
        ans = 'logLikelihood = {}'.format(loglikehood)
        print(ans)
        not_converge = np.abs(loglikehood - l_old) >= (10**-6 * np.abs(loglikehood) + 1e-16)
        l_old = loglikehood
    return mixture, post, loglikehood

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post, loglikehood = estep(X, mixture)
    mu = mixture.mu
    num, d = X.shape
    X_pred = np.copy(X)
    K = mu.shape[0]

    for n in np.arange(num):
        mask_flag = [i for i in np.arange(d) if X[n, i] == 0]
        # if there is any entry missing, get a predicted value
        if (len(mask_flag) > 0):
            predicted = np.matmul(post[n], mu)
            X_pred[n, mask_flag] = predicted[mask_flag]

    return X_pred