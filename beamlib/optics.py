"""
Linear optics: Twiss parameters, phase advance, normalised coordinates,
closed orbit and dispersion.

Conventions
-----------
* Twiss parameters (beta [m], alpha, gamma = (1 + alpha^2)/beta [1/m]) and
  the phase advance mu [rad].  The tune is Q = mu / (2 pi).
* A stable one-turn 2x2 matrix can always be written as

      M = cos(mu) I + sin(mu) J,     J = [[ alpha,  beta ],
                                          [-gamma, -alpha]]

* Courant-Snyder normalised coordinates (the ellipse becomes a circle):

      x_hat = x / sqrt(beta)
      px_hat = (alpha x + beta px) / sqrt(beta)

  In these coordinates one turn is a clockwise rotation by mu, and the
  single-particle invariant (emittance of that particle) is

      eps_i = x_hat^2 + px_hat^2 = gamma x^2 + 2 alpha x px + beta px^2 = 2 Jx

  where Jx is the action. eps_i is the area of the particle's ellipse / pi.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "twiss_from_matrix", "matrix_from_twiss", "transfer_matrix_from_twiss",
    "propagate_twiss", "twiss_along", "slice_elements", "cumulative_matrices",
    "ring_twiss", "normalise", "denormalise",
    "normalisation_matrix", "closed_orbit_linear", "dispersion_from_matrix",
    "one_turn_matrix", "tunes_from_matrix", "symplectic_form", "is_symplectic",
    "numerical_jacobian", "find_closed_orbit", "find_fixed_point",
]


# ---------------------------------------------------------------------------
# Twiss parameters <-> matrices
# ---------------------------------------------------------------------------
def twiss_from_matrix(M: np.ndarray) -> dict:
    """
    Extract (beta, alpha, gamma, mu, Q) from a 2x2 one-turn matrix.
    Raises ValueError if the motion is unstable (|Tr M| / 2 >= 1).
    mu is returned in [0, 2 pi) and Q = mu / 2 pi is the *fractional* tune.

    Parameters
    ----------
    M : np.ndarray
        Numpy array of shape (2, 2) representing a one-turn matrix.

    Returns
    -------
    dict{str: float}
        Dictionary with following keys: 'beta', 'alpha', 'gamma', 'mu', 'Q'
    """
    M = np.asarray(M, dtype=float)
    cos_mu = 0.5 * np.trace(M)
    if abs(cos_mu) >= 1.0:
        raise ValueError(f"Unstable motion: |Tr M|/2 = {abs(cos_mu):.4f} >= 1")
    # sign of sin(mu) is fixed by requiring beta > 0, i.e. sign(M12)
    sin_mu = np.sign(M[0, 1]) * np.sqrt(1.0 - cos_mu**2)
    beta = M[0, 1] / sin_mu
    alpha = (M[0, 0] - M[1, 1]) / (2.0 * sin_mu)
    gamma = -M[1, 0] / sin_mu
    mu = np.arctan2(sin_mu, cos_mu) % (2 * np.pi)
    return dict(beta=beta, alpha=alpha, gamma=gamma, mu=mu, Q=mu / (2 * np.pi))


def matrix_from_twiss(beta: float, alpha: float, mu: float) -> np.ndarray:
    """
    One-turn 2x2 matrix  M = cos(mu) I + sin(mu) J  for given Twiss and mu.
    
    Parameters
    ----------
    beta : float
        Twiss beta function [m].
    alpha : float
        Twiss alpha function [1].
    mu : float
        Phase advance per turn in [0, 2 pi).

    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) representing the one-turn map.
    """
    gamma = (1.0 + alpha**2) / beta
    J = np.array([[alpha, beta], [-gamma, -alpha]])
    return np.cos(mu) * np.eye(2) + np.sin(mu) * J


def transfer_matrix_from_twiss(beta1, alpha1, beta2, alpha2, dmu) -> np.ndarray:
    """
    2x2 transfer matrix from point 1 to point 2 given the Twiss at both
    points and the phase advance dmu between them.

    Parameters
    ----------
    beta1 : float
        Twiss beta function at first location [m].
    alpha1 : float
        Twiss alpha function at first location [1].
    beta2 : float
        Twiss beta function at second location [m].
    alpha2 : float
        Twiss alpha function at second location [1].

    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) representing the one-turn map.
    """
    s, c = np.sin(dmu), np.cos(dmu)
    sb = np.sqrt(beta2 / beta1)
    return np.array([
        [sb * (c + alpha1 * s), np.sqrt(beta1 * beta2) * s],
        [((alpha1 - alpha2) * c - (1 + alpha1 * alpha2) * s) / np.sqrt(beta1 * beta2),
         (c - alpha2 * s) / sb],
    ])


def propagate_twiss(M: np.ndarray, beta: float, alpha: float):
    """
    Propagate (beta, alpha) through the 2x2 matrix M.
    Returns (beta2, alpha2, dmu) where dmu is the phase advance in [0, 2 pi).
    Uses  B2 = M B1 M^T  with  B = [[beta, -alpha], [-alpha, gamma]].

    Parameters
    ----------
    M : np.ndarray
        Numpy array of shape (2, 2) representing transfer matrix.
    beta : float
        Twiss beta function [m].
    alpha : float
        Twiss alpha function [1].

    Returns
    -------
    float
        Propagated beta function [m].
    float
        Propagated alpha function [1].
    float
        Phase advance between location of input Twiss parameters and propagated 
        Twiss parameters.
    """
    gamma = (1 + alpha**2) / beta
    B1 = np.array([[beta, -alpha], [-alpha, gamma]])
    B2 = M @ B1 @ M.T
    dmu = np.arctan2(M[0, 1], M[0, 0] * beta - M[0, 1] * alpha) % (2 * np.pi)
    return B2[0, 0], -B2[0, 1], dmu


def slice_elements(elements, n_slices: int = 10):
    """
    Cut thick drifts, quads and bends into ``n_slices`` equal pieces.
    Returns a list of (element_index, piece).  Other elements are kept whole.
    Useful to get smooth optics functions along s.

    Parameters
    ----------
    elements : list
        List of beam elements from beamlib.
    n_sliced : int
        Number of slices to slice drifts, quads and bends into.

    Returns
    -------
    list[tuple]
        List of tuples that each contain the index of the element the slice 
        belongs to and the slice itself.
    """
    from .elements import Drift, Quadrupole, SectorBend
    out = []
    for i, el in enumerate(elements):
        if isinstance(el, Drift) and el.L > 0:
            out += [(i, Drift(el.L / n_slices, name=el.name))] * n_slices
        elif isinstance(el, Quadrupole):
            out += [(i, Quadrupole(el.k, el.L / n_slices, name=el.name))] * n_slices
        elif isinstance(el, SectorBend):
            out += [(i, SectorBend(el.L / n_slices, el.rho, name=el.name))] * n_slices
        else:
            out.append((i, el))
    return out


def cumulative_matrices(elements, n_slices: int = 10, delta: float = 0.0):
    """
    Matrices from the start of the line to many points along it.
    Returns (s, elem, Ms): positions s [m] (first entry 0), the index of the
    element each point is the exit of (-1 for the start), and an array of
    5x5 matrices with Ms[j] = matrix from s = 0 to s[j].

    Parameters
    ----------
    elements : list
        List of beamlib beam elements.
    n_slices : int
        Number of slices to slice each drift, quad and bend into.
    delta : float
        Momentum offset [1] of particle for which transfer matrices are computed.

    Returns
    -------
    np.ndarray
        Numpy array of `s` locations.
    np.ndarray
        Numpy array of (sliced) elements.
    np.ndarray
        Numpy array of transfer matrices of shape (5, 5).
    """
    s, elem, Ms = [0.0], [-1], [np.eye(5)]
    for i, pc in slice_elements(elements, n_slices):
        s.append(s[-1] + pc.length)
        elem.append(i)
        Ms.append(np.matmul(pc.matrix(delta), Ms[-1]))
    return np.array(s), np.array(elem), np.array(Ms)


def twiss_along(elements, beta_x, alpha_x, beta_y, alpha_y, delta=0.0,
                n_slices: int = 10):
    """
    Propagate Twiss parameters through a list of elements.
    Thick elements are cut into ``n_slices`` slices to get smooth curves.
    Returns a dict of arrays: s, betx, alfx, mux, bety, alfy, muy, and the
    index of the element each point belongs to (``elem``).  The values are
    *at the exit* of each slice; the first entry is the start of the line.
    The phase advances mux, muy [rad] accumulate (they are not wrapped).

    Parameters
    ----------
    elements : list
        List of beamlib elements.
    beta_x : float
        Horizontal Twiss beta function [m].
    alha_x : float
        Horizontal Twiss alpha function [1].
    beta_x : float
        Horizontal Twiss beta function [m].
    alha_x : float
        Horizontal Twiss alpha function [1].
    delta : float
        Momentum offset [1] of particle for which transfer matrices are computed.

    Returns
    -------
    dict{str: np.ndarray}
        Dictionary with keys s, betx, alfx, mux, bety, alfy, muy and element 
        index, and values of corresponding arrays as a function of s.
    """
    out = {k: [v] for k, v in dict(s=0.0, betx=beta_x, alfx=alpha_x, mux=0.0,
                                   bety=beta_y, alfy=alpha_y, muy=0.0, elem=-1).items()}
    for i, pc in slice_elements(elements, n_slices):
        M = pc.matrix(delta)
        bx, ax, dmx = propagate_twiss(M[:2, :2], out["betx"][-1], out["alfx"][-1])
        by, ay, dmy = propagate_twiss(M[2:4, 2:4], out["bety"][-1], out["alfy"][-1])
        out["s"].append(out["s"][-1] + pc.length)
        out["betx"].append(bx); out["alfx"].append(ax)
        out["bety"].append(by); out["alfy"].append(ay)
        # one slice never advances the phase by more than pi; avoid 2 pi wraps
        dmx = dmx if dmx < np.pi else dmx - 2 * np.pi
        dmy = dmy if dmy < np.pi else dmy - 2 * np.pi
        out["mux"].append(out["mux"][-1] + dmx)
        out["muy"].append(out["muy"][-1] + dmy)
        out["elem"].append(i)
    return {k: np.array(v) for k, v in out.items()}


def ring_twiss(elements, delta: float = 0.0, n_slices: int = 10):
    """
    Periodic Twiss functions and dispersion around a ring (convenience).
    Returns the dict of :func:`twiss_along` plus ``dx``, ``dpx`` (periodic
    dispersion D, D' [m], [-]) and the total tunes ``Qx``, ``Qy`` (with
    integer part).

    Parameters
    ----------
    elements : list
        List of beamlib elements.
    delta : float
        Momentum offset [1] of particle for which transfer matrices are computed.
    n_slices : int
        Number of slices to slice each drift, quad and bend into.

    Returns
    -------
    dict{str: np.ndarray}
        Dictionary with keys s, betx, alfx, mux, bety, alfy, muy, dx, dpx, Qx, Qy 
        and element index, and values of corresponding arrays as a function of s.
    """
    M = one_turn_matrix(elements, delta)
    tx, ty = twiss_from_matrix(M[:2, :2]), twiss_from_matrix(M[2:4, 2:4])
    tw = twiss_along(elements, tx["beta"], tx["alpha"], ty["beta"], ty["alpha"],
                     delta=delta, n_slices=n_slices)
    D0, Dp0 = dispersion_from_matrix(M)
    _, _, Ms = cumulative_matrices(elements, n_slices, delta)
    disp = Ms @ np.array([D0, Dp0, 0.0, 0.0, 1.0])
    tw["dx"], tw["dpx"] = disp[:, 0], disp[:, 1]
    tw["Qx"], tw["Qy"] = tw["mux"][-1] / (2 * np.pi), tw["muy"][-1] / (2 * np.pi)
    return tw


# ---------------------------------------------------------------------------
# Courant-Snyder normalised coordinates
# ---------------------------------------------------------------------------
def normalisation_matrix(beta: float, alpha: float) -> np.ndarray:
    """
    2x2 matrix N with (x_hat, p_hat) = N (x, px).

    Parameters
    ----------
    beta_x : float
        Twiss beta function [m].
    alha_x : float
        Twiss alpha function [1].
    
    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) to normalise coordinates.
    """
    sb = np.sqrt(beta)
    return np.array([[1.0 / sb, 0.0], [alpha / sb, sb]])


def normalise(x, px, beta: float, alpha: float):
    """
    Physical (x, px) -> normalised (x_hat, p_hat).

    Parameters
    ----------
    x : np.ndarray
        Numpy array of position coordinates [m].
    px : np,ndarray
        Numpy array of angle coordinates [rad].
    beta : float
        Twiss beta function [m].
    alha : float
        Twiss alpha function [1].
    
    Returns
    -------
    np.ndarray
        Numpy array of normalised position coordinates [sqrt(m)].
    np.ndarray
        Numpy array of normalised angle coordinates [sqrt(m)].
    """
    sb = np.sqrt(beta)
    return np.asarray(x) / sb, (alpha * np.asarray(x) + beta * np.asarray(px)) / sb


def denormalise(x_hat, p_hat, beta: float, alpha: float):
    """
    Normalised (x_hat, p_hat) -> physical (x, px).
    
    Parameters
    ----------
    x_hat : np.ndarray
        Numpy array of normalised position coordinates [sqrt(m)].
    px_hat : np,ndarray
        Numpy array of normalised angle coordinates [sqrt(m)].
    beta : float
        Twiss beta function [m].
    alha : float
        Twiss alpha function [1].
    
    Returns
    -------
    np.ndarray
        Numpy array of position coordinates [m].
    np.ndarray
        Numpy array of angle coordinates [rad].
    """
    sb = np.sqrt(beta)
    x = sb * np.asarray(x_hat)
    return x, (np.asarray(p_hat) - alpha * np.asarray(x_hat)) / sb


# ---------------------------------------------------------------------------
# Lattices: one-turn matrix, tunes, closed orbit, dispersion
# ---------------------------------------------------------------------------
def one_turn_matrix(elements, delta: float = 0.0) -> np.ndarray:
    """
    5x5 one-turn matrix: product of the element matrices (first element
    acts first, so it is the right-most factor).

    Parameters
    ----------
    elements : list
        List of beamlib elements.
    delta : float
        Momentum offset [1] of particle for which transfer matrices are computed.

    Returns
    -------
    np.ndarray
        Numpy array of shape (5, 5) representing the one-turn matrix.
    """
    M = np.eye(5)
    for el in elements:
        M = np.matmul(el.matrix(delta), M)
    return M


def tunes_from_matrix(M: np.ndarray):
    """
    Fractional tunes (Qx, Qy) of an uncoupled 4x4 (or 5x5) matrix.
    
    Parameters
    ----------
    M : np.ndarray
        Numpy array of shape (4, 4) or (5, 5) representing the one-turn matrix.

    Returns
    -------
    float
        Horizontal fractional tune.
    float
        Vertical fractional tune.
    """
    return (twiss_from_matrix(M[:2, :2])["Q"], twiss_from_matrix(M[2:4, 2:4])["Q"])


def closed_orbit_linear(M5: np.ndarray, delta: float) -> np.ndarray:
    """
    Closed orbit (x, px, y, py) for momentum deviation delta.
    The closed orbit is the fixed point  z = M4 z + d delta, so
    z = (I - M4)^(-1) d delta, where d = M5[:4, 4] is the dispersion column.

    Parameters
    ----------
    M5 : np.ndarray
        Numpy array of shape (5, 5) representing the one-turn matrix.
    delta : float
        Momentum offset [1] of particle.

    Returns
    -------
    np.ndarray
        (x, px, y, py) of closed orbit.
    """
    M4, d = M5[:4, :4], M5[:4, 4]
    return np.linalg.solve(np.eye(4) - M4, d * delta)


def dispersion_from_matrix(M5: np.ndarray):
    """
    Periodic dispersion (D, D') at the start of the one-turn matrix.
    
    Parameters
    ----------
    M5 : np.ndarray
        Numpy array of shape (5, 5) representing the one-turn matrix.

    Returns
    -------
    float
        Dispersion D at start of one-turn matrix.
    float
        Derivative of dispersion D' at start of one-turn matrix.
    """
    co = closed_orbit_linear(M5, 1.0)
    return co[0], co[1]


# ---------------------------------------------------------------------------
# Symplecticity and nonlinear maps
# ---------------------------------------------------------------------------
def symplectic_form(n: int = 4) -> np.ndarray:
    """
    The matrix S = diag(s, s, ...) with s = [[0, 1], [-1, 0]] (n even).
    
    Parameters
    ----------
    n : int
        Number of dimensions.
    
    Returns
    -------
    np.ndarray
        Symplectic matrix.
    """
    S = np.zeros((n, n))
    for i in range(0, n, 2):
        S[i, i + 1], S[i + 1, i] = 1.0, -1.0
    return S


def is_symplectic(M: np.ndarray, tol: float = 1e-9) -> bool:
    """
    True if M^T S M = S (for 2x2 this is just det M = 1).
    
    Parameters
    ----------
    M : np.ndarray
        Numpy array representing transfer matrix.
    tol : float
        Tolerance for symplecticity to be satisfied.

    Returns
    -------
    bool
        True is matrix is symplectic, False if not.
    """
    n = M.shape[0] - (M.shape[0] % 2)   # ignore the delta row of a 5x5 matrix
    M = M[:n, :n]
    S = symplectic_form(n)
    return bool(np.allclose(M.T @ S @ M, S, atol=tol))


def numerical_jacobian(one_turn_map, z0, eps: float = 1e-7) -> np.ndarray:
    """
    Jacobian dz_out/dz_in of a map at the point z0 (central differences).
    ``one_turn_map`` takes a bunch of shape (N, n) and returns the same shape.
    All 2n displaced copies are tracked together as one bunch.
    """
    z0 = np.asarray(z0, dtype=float)
    n = len(z0)
    bunch = np.repeat(z0[None, :], 2 * n, axis=0)
    for i in range(n):
        bunch[2 * i, i] += eps
        bunch[2 * i + 1, i] -= eps
    out = one_turn_map(bunch)
    return ((out[0::2] - out[1::2]) / (2 * eps)).T


def find_closed_orbit(one_turn_map, delta: float = 0.0, guess=None,
                      tol: float = 1e-13, max_iter: int = 50) -> np.ndarray:
    """
    Closed orbit of a (possibly nonlinear) map by Newton iteration.
    Works on 5-column bunches (x, px, y, py, delta); delta is held fixed.
    Returns the 4-vector (x, px, y, py).
    """
    z = np.zeros(4) if guess is None else np.array(guess, dtype=float)

    def f(zz):  # map restricted to the transverse coordinates
        b = np.column_stack([zz, np.full(len(zz), delta)])
        return one_turn_map(b)[:, :4]

    for _ in range(max_iter):
        J = numerical_jacobian(f, z)
        r = f(z[None, :])[0] - z
        dz = np.linalg.solve(J - np.eye(4), -r)
        z = z + dz
        if np.max(np.abs(dz)) < tol:
            break
    return z


def find_fixed_point(one_turn_map, z0, period: int = 1, dims=(0, 1),
                     tol: float = 1e-14, max_iter: int = 60):
    """
    Fixed point of ``period`` turns of a map, by Newton iteration.

    Parameters
    ----------
    one_turn_map : callable bunch (N, 4) -> bunch (N, 4)
    z0 : initial guess (x, px, y, py)
    period : int
        Look for points that come back after ``period`` turns (e.g. 3 for
        the third-order resonance).
    dims : indices that are solved for; the other coordinates are kept at
        their values in ``z0``.  Default (0, 1): horizontal plane only,
        which is exact when y = py = 0 is invariant (normal multipoles).

    Returns
    -------
    z : array (4,) the fixed point, or None if Newton did not converge.
    trace : float, trace of the ``period``-turn Jacobian in ``dims``.
        |trace| < 2 (for a 2D map): stable (elliptic, centre of an island);
        |trace| > 2: unstable (hyperbolic, on a separatrix).
    """
    dims = list(dims)
    z = np.array(z0, dtype=float)

    def f(sub):  # period-turn map restricted to the chosen coordinates
        b = np.repeat(z[None, :], len(sub), axis=0)
        b[:, dims] = sub
        for _ in range(period):
            b = one_turn_map(b)
        return b[:, dims]

    with np.errstate(all="ignore"):
        for _ in range(max_iter):
            sub = z[dims]
            r = f(sub[None, :])[0] - sub
            if not np.all(np.isfinite(r)):
                return None, np.nan
            J = numerical_jacobian(f, sub, eps=1e-9)
            try:
                d = np.linalg.solve(J - np.eye(len(dims)), -r)
            except np.linalg.LinAlgError:
                return None, np.nan
            z[dims] = sub + d
            if np.max(np.abs(d)) < tol:
                break
        sub = z[dims]
        if not np.all(np.isfinite(sub)) or np.max(np.abs(sub)) > 1.0:
            return None, np.nan              # diverged (|x| > 1 m is not physical here)
        if np.max(np.abs(f(sub[None, :])[0] - sub)) > 1e-10:
            return None, np.nan
        return z, float(np.trace(numerical_jacobian(f, sub, eps=1e-9)))
