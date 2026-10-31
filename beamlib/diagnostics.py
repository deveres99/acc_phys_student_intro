"""
Diagnostics: invariants, action-angle variables, tunes, beam moments,
emittances and a simple chaos indicator.

The two emittances (keep them apart!)
-------------------------------------
* Single-particle emittance (Courant-Snyder invariant) of particle i::

      eps_i = gamma x^2 + 2 alpha x px + beta px^2 = x_hat^2 + p_hat^2 = 2 J_i

  It is the area / pi of the ellipse that this one particle moves on, and it
  uses the *lattice* Twiss parameters (beta, alpha, gamma).

* RMS emittance of a beam (a property of the particle distribution)::

      eps_rms = sqrt(det Sigma),  Sigma = [[<x x>,  <x px>],
                                           [<px x>, <px px>]]

  with centred second moments.  It uses no lattice information at all.

For a Gaussian beam matched to the lattice, eps_i is exponentially
distributed with mean <eps_i> = 2 eps_rms (so <J> = eps_rms), and the
1-sigma ellipse (eps_i <= eps_rms) contains 1 - exp(-1/2) ~ 39 % of the
particles.  In general <eps_i>/2 >= eps_rms, with equality only for a matched,
centred beam.

Tunes
-----
All tune functions take turn-by-turn data with turns along axis 0 and
return the *fractional* tune in [0, 1).  Passing the complex signal
z = x_hat - i p_hat removes the Q <-> 1 - Q ambiguity of a real signal.
"""
from __future__ import annotations

import numpy as np

from .optics import normalise

__all__ = [
    "cs_invariant", "action_angle", "beam_sigma", "rms_emittance",
    "tune_phase_advance", "tune_fft", "tune_naff", "complex_signal",
    "chaos_indicator",
]


# ---------------------------------------------------------------------------
# Invariants and action-angle variables
# ---------------------------------------------------------------------------
def cs_invariant(x, px, beta: float, alpha: float):
    """
    Courant-Snyder invariant eps_i = gamma x^2 + 2 alpha x px + beta px^2.
    Equal to the area/pi of the particle's phase-space ellipse, and to 2 J.
    Works element-wise on arrays.

    Parameters
    ----------
    x : float | np.ndarray
        Particle position coordinate(s) [m].
    px : float | np.ndarray
        Particle angle coordinate(s) [rad].
    beta : float
        Twiss beta function [m].
    alpha : float
        Twiss alpha function [1].

    Returns
    -------
    float | np.ndarray
        Value of Courant-Snyder invariant(s).
    """
    gamma = (1.0 + alpha**2) / beta
    x, px = np.asarray(x), np.asarray(px)
    return gamma * x**2 + 2 * alpha * x * px + beta * px**2


def action_angle(x, px, beta: float, alpha: float):
    """
    Action J = eps_i / 2 and angle phi in [0, 2 pi).
    Defined by  x_hat = sqrt(2J) cos(phi),  p_hat = -sqrt(2J) sin(phi),
    so that phi *increases* by mu every turn.

    Parameters
    ----------
    x : float | np.ndarray
        Particle position coordinate(s) [m].
    px : float | np.ndarray
        Particle angle coordinate(s) [rad].
    beta : float
        Twiss beta function [m].
    alpha : float
        Twiss alpha function [1].

    Returns
    -------
    float | np.ndarray
        Value of action(s).
    float | np.ndarray
        Value of angle(s).
    """
    xh, ph = normalise(x, px, beta, alpha)
    J = 0.5 * (xh**2 + ph**2)
    phi = np.arctan2(-ph, xh) % (2 * np.pi)
    return J, phi


def complex_signal(x, px, beta: float, alpha: float):
    """
    z = x_hat - i p_hat = sqrt(2J) exp(i phi): rotates as exp(2 pi i Q n).
    
    Parameters
    ----------
    x : float | np.ndarray
        Particle position coordinate(s) [m].
    px : float | np.ndarray
        Particle angle coordinate(s) [rad].
    beta : float
        Twiss beta function [m].
    alpha : float
        Twiss alpha function [1].

    Returns
    -------
    complex | np.ndarray
        Complex Courant-Snyder coordinate(s).
    """
    xh, ph = normalise(x, px, beta, alpha)
    return xh - 1j * ph


# ---------------------------------------------------------------------------
# Beam moments
# ---------------------------------------------------------------------------
def beam_sigma(particles, rows=(0, 1)):
    """
    Covariance (Sigma) matrix of the chosen rows (centred moments).
    ``beam_sigma(p)`` gives the 2x2 horizontal Sigma matrix,
    ``beam_sigma(p, (0, 1, 2, 3))`` the full 4x4 one.  NaNs (lost particles)
    are ignored.

    Parameters
    ----------
    particles : np.ndarray
        Numpy array of shape (n, N), where n can be 2, 4, 5.
    rows : tuple
        Tuple of rows to index out.
    
    Returns
    -------
    np.ndarray
        Numpy array representing the covariance matrix.
    """
    p = np.asarray(particles)[list(rows), :]
    p = p[np.isfinite(p).all(axis=0)]
    d = p - p.mean(axis=1)
    return d.T @ d / p.shape[1]


def rms_emittance(x, px):
    """
    RMS emittance sqrt(det Sigma) of a distribution (centred moments).
    Accepts arrays of shape (N,) or (n_turns, N): the emittance is computed
    over the last axis (the particles), giving one value per turn.

    Parameters
    ----------
    x : np.ndarray
        Numpy array of position coordinates.
    px : np.ndarray
        Numpy array of angle coordinates.

    Returns
    -------
    float
        RMS emittance.
    """
    x, px = np.asarray(x, dtype=float), np.asarray(px, dtype=float)
    dx = x - np.nanmean(x, axis=-1, keepdims=True)
    dp = px - np.nanmean(px, axis=-1, keepdims=True)
    sxx = np.nanmean(dx * dx, axis=-1)
    spp = np.nanmean(dp * dp, axis=-1)
    sxp = np.nanmean(dx * dp, axis=-1)
    return np.sqrt(sxx * spp - sxp**2)


# ---------------------------------------------------------------------------
# Tunes
# ---------------------------------------------------------------------------
def tune_phase_advance(x, px, beta: float, alpha: float):
    """
    Tune from the average rotation angle per turn in normalised phase space.
    x, px : arrays with turns along axis 0, shape (n_turns,) or (n_turns, N).
    Exact for linear motion; for nonlinear motion it gives the average tune.

    Parameters
    ----------
    x : np.ndarray
        Numpy array of position coordinates.
    px : np.ndarray
        Numpy array of angle coordinates.

    Returns
    -------
    float
        Tunes calculated from average phase advance.
    """
    _, phi = action_angle(x, px, beta, alpha)
    dphi = np.diff(phi, axis=0) % (2 * np.pi)
    return np.mean(dphi, axis=0) / (2 * np.pi)


def tune_fft(signal, window: bool = True):
    """
    Tune from the FFT peak, refined by interpolation.

    Parameters
    ----------
    signal : np.ndarray
        Real or complex array, turns along axis 0.  For a real signal
        only tunes in [0, 0.5] can be told apart (Q and 1 - Q give the same
        x(n)), so the result is folded into [0, 0.5].
    window : bool
        If True, apply a Hann window, which makes the interpolation accurate
        (error ~ 1e-5 / N for a clean signal instead of ~ 1/N).

    Returns
    -------
    np.ndarray
        Array of particle tunes for each particle.
    """
    s = np.asarray(signal)
    is_complex = np.iscomplexobj(s)
    n = s.shape[0]
    s = s - s.mean(axis=0)
    w = np.hanning(n) if window else np.ones(n)
    w = w.reshape((n,) + (1,) * (s.ndim - 1))
    A = np.abs(np.fft.fft(s * w, axis=0))
    if not is_complex:
        A[n // 2 + 1:] = 0.0              # negative frequencies are a mirror image
    k = np.argmax(A, axis=0)
    # three-point interpolation of the peak position
    a0 = np.take_along_axis(A, ((k - 1) % n)[None], axis=0)[0]
    a1 = np.take_along_axis(A, k[None], axis=0)[0]
    a2 = np.take_along_axis(A, ((k + 1) % n)[None], axis=0)[0]
    if window:   # exact for a Hann window and a pure sinusoid near the bin
        with np.errstate(invalid="ignore", divide="ignore"):
            shift = np.where(a2 > a0, (2 * a2 - a1) / (a1 + a2), -(2 * a0 - a1) / (a1 + a0))
    else:        # parabolic interpolation (rough)
        with np.errstate(invalid="ignore", divide="ignore"):
            shift = 0.5 * (a0 - a2) / (a0 - 2 * a1 + a2)
    q = ((k + np.nan_to_num(shift)) / n) % 1.0
    return np.where(np.isfinite(s).all(axis=0), q, np.nan)   # lost particles -> NaN


def tune_naff(signal, n_iter: int = 45):
    """
    Refined tune: maximise |sum_n z_n w_n exp(-2 pi i Q n)| near the FFT peak.
    This is the core idea of NAFF (Numerical Analysis of Fundamental
    Frequencies), fully vectorised over particles.  Precision for a clean
    signal of N turns is ~1e-10 or better (limited by round-off).
    Use the complex signal z = x_hat - i p_hat for best results.
    """
    s = np.asarray(signal)
    one = s.ndim == 1
    if one:
        s = s[:, None]
    n = s.shape[0]
    s = s - s.mean(axis=0)
    w = np.hanning(n)[:, None]
    turns = np.arange(n)[:, None]
    q0 = tune_fft(s if np.iscomplexobj(s) else s, window=True)

    def amp(q):  # |windowed Fourier integral| at frequency q (one per particle)
        return np.abs(np.sum(s * w * np.exp(-2j * np.pi * q[None, :] * turns), axis=0))

    # golden-section search in [q0 - 1/n, q0 + 1/n], vectorised over particles.
    # Each step shrinks the interval by 0.618 and needs one new evaluation.
    g = (np.sqrt(5) - 1) / 2
    a, b = q0 - 1.0 / n, q0 + 1.0 / n
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = amp(c), amp(d)
    for _ in range(n_iter):
        left = fc > fd                  # maximum lies in [a, d]
        a = np.where(left, a, c)
        b = np.where(left, d, b)
        new = np.where(left, b - g * (b - a), a + g * (b - a))
        f_new = amp(new)
        c, d, fc, fd = (np.where(left, new, d), np.where(left, c, new),
                        np.where(left, f_new, fd), np.where(left, fc, f_new))
    q = (0.5 * (a + b)) % 1.0
    if not np.iscomplexobj(s):
        q = np.where(q > 0.5, 1.0 - q, q)
    q = np.where(np.isfinite(s).all(axis=0), q, np.nan)   # lost particles -> NaN
    return q[0] if one else q


def chaos_indicator(signal):
    """
    Frequency-map chaos indicator: log10 |Q(first half) - Q(second half)|.
    Regular (non-chaotic) motion has a well-defined tune, so both halves give
    the same Q and the indicator is very negative (< -6 or so).  Chaotic
    motion has a wandering tune: the indicator approaches -2 ... -1.
    signal : complex turn-by-turn signal, turns along axis 0.
    """
    s = np.asarray(signal)
    h = s.shape[0] // 2
    q1, q2 = tune_naff(s[:h]), tune_naff(s[h:2 * h])
    dq = np.abs(q1 - q2)
    dq = np.minimum(dq, 1 - dq)
    with np.errstate(divide="ignore"):
        return np.log10(np.maximum(dq, 1e-16))
