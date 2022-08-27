"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

from scipy.stats import multivariate_normal


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    mu = mixture.mu
    var = mixture.var
    pj = mixture.p

    K = len(pj)
    n, d = X.shape
    pj_i = np.zeros((n, K))

    for k in range(K):
        pj_i[:, k] = pj[k] * multivariate_normal.pdf(X, mean=mu[k,:], cov=var[k])
    px = np.sum(pj_i, axis=1)
    pj_i = pj_i / px.reshape(n,1)

    # calculate log-likelihood
    val = np.log(px)
    ll = np.sum(val)

    return (pj_i, ll)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

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
    ll_old = None
    ll_new = None
    while (ll_old is None or ll_new is None or ll_new - ll_old <= 1.0e-6 * abs(ll_new)):
        ll_old = ll_new
        post, ll_new = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll_new

