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
    logp_j = np.transpose(np.log(p))

    # 1. build nn
    lj_x = []
    loglikelihood = 0
    for i in np.arange(num):
        nn = np.array([norm.pdf(X[i,cu[i]], mean=mu[j,cu[i]], cov=(var[j])) for j in np.arange(K)])
        # for numerical stability, use logsumexp and take out the average
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
        not_converge = true
    return mixture, post, loglikehood

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
