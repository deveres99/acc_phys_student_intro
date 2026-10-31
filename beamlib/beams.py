"""
Particle distributions.

A Gaussian beam is fully described by its mean (usually zero) and its
covariance matrix Sigma.  A beam is *matched* to the lattice when its
Sigma matrix has the same shape as the lattice ellipse:

    Sigma = eps_rms * [[ beta, -alpha],
                       [-alpha, gamma]]

so that sqrt(det Sigma) = eps_rms.
"""
from __future__ import annotations

import numpy as np

__all__ = ["matched_sigma", "gaussian_from_sigma", "matched_gaussian"]


def matched_sigma(eps_rms: float, beta: float, alpha: float) -> np.ndarray:
    """
    2x2 Sigma matrix of a beam matched to Twiss (beta, alpha).
    """
    gamma = (1 + alpha**2) / beta
    return eps_rms * np.array([[beta, -alpha], [-alpha, gamma]])


def gaussian_from_sigma(Sigma, n: int, mean=None, rng=None) -> np.ndarray:
    """
    n samples of a Gaussian with covariance Sigma (any dimension).
    Uses the Cholesky factor L (Sigma = L L^T): if u is a vector of
    independent unit Gaussians, then L u has covariance Sigma.
    """
    rng = np.random.default_rng(rng)
    Sigma = np.asarray(Sigma, dtype=float)
    L = np.linalg.cholesky(Sigma)
    u = rng.standard_normal((n, len(Sigma)))
    out = u @ L.T
    if mean is not None:
        out += np.asarray(mean)
    return out


def matched_gaussian(n: int, epsx: float, betx: float, alfx: float,
                     epsy: float = 0.0, bety: float = 1.0, alfy: float = 0.0,
                     delta_rms: float = 0.0, rng=None) -> np.ndarray:
    """
    Bunch (n, 4) — or (n, 5) if delta_rms > 0 — matched in both planes.
    """
    rng = np.random.default_rng(rng)
    cols = [gaussian_from_sigma(matched_sigma(epsx, betx, alfx), n, rng=rng)]
    if epsy > 0:
        cols.append(gaussian_from_sigma(matched_sigma(epsy, bety, alfy), n, rng=rng))
    else:
        cols.append(np.zeros((n, 2)))
    if delta_rms > 0:
        cols.append(delta_rms * rng.standard_normal((n, 1)))
    return np.hstack(cols)
