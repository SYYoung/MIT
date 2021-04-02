"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal as norm


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
    raise NotImplementedError


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
    pj_x, loglikehood = estep(X, mixture)
    return mixture, pj_x, loglikehood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
