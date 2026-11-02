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

    Parameters
    ----------
    eps_rms : float
        RMS emittance [m].
    beta : float
        Twiss beta function [m].
    alpha : float
        Twiss alpha function [1].

    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) representing sigma matrix.
    """
    gamma = (1 + alpha**2) / beta
    return eps_rms * np.array([[beta, -alpha], [-alpha, gamma]])


def gaussian_from_sigma(Sigma, n: int, mean=None, rng=None) -> np.ndarray:
    """
    n samples of a Gaussian with covariance Sigma (any dimension d).
    Uses the Cholesky factor L (Sigma = L L^T): if u is a vector of
    independent unit Gaussians, then L u has covariance Sigma.

    Parameters
    ----------
    Sigma : np.ndarray
        Covariance matrix of shape (d, d).
    n : int
        Number of particles.
    mean : np.ndarray
        Mean vector of length d. Default is zero.
    rng : int | np.random.Generator
        Seed or random generator.

    Returns
    -------
    np.ndarray
        Numpy array of shape (d, n): one row per coordinate, one column per particle.
    """
    rng = np.random.default_rng(rng)
    Sigma = np.asarray(Sigma, dtype=float)
    L = np.linalg.cholesky(Sigma)
    u = rng.standard_normal((len(Sigma), n))
    out = L @ u
    if mean is not None:
        out += np.asarray(mean, dtype=float)[:, None]
    return out


def matched_gaussian(n: int, epsx: float, betx: float, alfx: float,
                     epsy: float = 0.0, bety: float = 1.0, alfy: float = 0.0,
                     delta_rms: float = 0.0, rng=None) -> np.ndarray:
    """
    Gaussian bunch matched in both planes.

    Parameters
    ----------
    n : int
        Number of particles.
    epsx : float
        Horizontal emittance [m].
    betx : float
        Horizontal Twiss beta function [m].
    alfx : float
        Horizontal Twiss alpha function [1].
    epsy : float
        Vertical emittance [m].
    bety : float
        Vertical Twiss beta function [m].
    alfy : float
        Vertical Twiss alpha function [1].
    delta_rms : float
        RMS spread of momentum offset [1].
    rng : int | np.random.Generator
        Seed or random generator.

    Returns
    -------
    np.ndarray
        Numpy array of shape (4, n) with rows (x, px, y, py), or (5, n) with
        delta as the last row if delta_rms > 0. If epsy is 0 the vertical
        rows are zero.
    """
    rng = np.random.default_rng(rng)
    rows = [gaussian_from_sigma(matched_sigma(epsx, betx, alfx), n, rng=rng)]
    if epsy > 0:
        rows.append(gaussian_from_sigma(matched_sigma(epsy, bety, alfy), n, rng=rng))
    else:
        rows.append(np.zeros((2, n)))
    if delta_rms > 0:
        rows.append(delta_rms * rng.standard_normal((1, n)))
    return np.vstack(rows)
