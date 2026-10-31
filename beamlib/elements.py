"""
Beam-line elements: linear transfer matrices and thin multipole kicks.

Conventions used everywhere in ``beamlib``
------------------------------------------
* Phase-space coordinates of one particle are ``(x, px, y, py, delta)``:

  - ``x, y``   transverse positions [m]
  - ``px, py`` transverse angles  x' = dx/ds, y' = dy/ds [rad].
    We use the *small-angle approximation*: px = x' (no difference between
    angle and normalised momentum).
  - ``delta = (p - p0)/p0`` relative momentum deviation [dimensionless].

* A *bunch* of N particles is a NumPy array of shape ``(4, N)``
  (columns x, px, y, py) or ``(5, N)`` (with delta as the last column).
  All tracking is vectorised over particles: no Python loops over particles.

* ``k`` is the normalised quadrupole gradient [1/m^2] (MAD-X ``k1``).
  ``k > 0`` focuses in x and defocuses in y.

* Thin multipole kick, same convention as MAD-X and Xsuite
  (n = 0 dipole, 1 quadrupole, 2 sextupole, 3 octupole)::

    Δpx - i Δpy = - Σ_n (knl[n] + i ksl[n]) / n!  (x + i y)^n

  ``knl[n]`` is the integrated normal strength [1/m^n], ``ksl[n]`` the skew one.
  With this sign convention ``knl[1] > 0`` is a focusing quadrupole in x.

* Chromatic effects: since px = x', a magnet deflects a particle with
  momentum p0(1+delta) by an angle 1/(1+delta) times smaller.
  So when the bunch has a delta column, every *magnetic* kick or focusing
  strength is divided by (1 + delta). Dipole bending is the exception: it is
  kept linear and enters through the dispersion column of the 5x5 matrix.
  (Xsuite uses canonical momenta, where the same physics appears in the
  drifts instead; the resulting tunes and chromaticities are the same.)

Matrices
--------
2x2 matrices act on (x, px) or (y, py).  ``block_4d`` builds the 4x4 matrix
acting on (x, px, y, py).  ``matrix_5d`` adds the delta row/column::

        [ Mx   0   d ]        d = (R16, R26): position and angle picked up
    M = [ 0    My  0 ]            by a particle with delta = 1
        [ 0    0   1 ]

"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial

import numpy as np

__all__ = [
    "drift", "thin_quad", "thick_quad", "sector_dipole", "sector_dipole_dispersion",
    "block_4d", "matrix_5d", "apply_matrix", "thin_multipole",
    "Drift", "Quadrupole", "SectorBend", "ThinMultipole", "LinearMap", "Marker",
]


# ---------------------------------------------------------------------------
# 2x2 transfer matrices for one plane
# ---------------------------------------------------------------------------
def drift(L: float) -> np.ndarray:
    """
    2x2 matrix of a field-free drift of length L [m].

    Parameters
    ----------
    L : float
        Length of drift [m].

    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) representing the drift transfer matrix in 2D.
    """
    return np.array([[1.0, L], [0.0, 1.0]])


def thin_quad(kl: float) -> np.ndarray:
    """
    2x2 matrix of a thin quadrupole with integrated strength kl = k*L [1/m].
    kl > 0 focuses (focal length f = 1/kl), kl < 0 defocuses.
    For the vertical plane of a quadrupole use ``thin_quad(-kl)``.

    Parameters
    ----------
    kl : float
        Integrated normalised normal quadrupole strength [1/m].
    
    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) representing the thin lense approximation of 
        a normal quadrupole transfer matrix in 2D.
    """
    return np.array([[1.0, 0.0], [-kl, 1.0]])


def thick_quad(k: float, L: float) -> np.ndarray:
    """2x2 matrix of a thick quadrupole (or any element with x'' + k x = 0).

    Parameters
    ----------
    k : float
        Focusing strength [1/m^2]. k > 0 focusing, k < 0 defocusing.
        k = 0 gives back a drift (the limit is handled smoothly).
    L : float
        Length of quadrupole [m].

    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) representing the normal quadrupole transfer 
        matrix in 2D.
    """
    # Smooth formulas that work for both signs of k and for k -> 0:
    #   C = cos(sqrt(k)*L),  S = sin(sqrt(k)*L)/sqrt(k),  C' = -k S
    # For k < 0, cos/sin of an imaginary argument become cosh/sinh.
    if abs(k) * L**2 < 1e-10:            # Taylor expansion around k = 0
        C = 1.0 - k * L**2 / 2
        S = L - k * L**3 / 6
    elif k > 0:
        sk = np.sqrt(k)
        C = np.cos(sk * L)
        S = np.sin(sk * L) / sk
    else:
        sk = np.sqrt(-k)
        C = np.cosh(sk * L)
        S = np.sinh(sk * L) / sk
    return np.array([[C, S], [-k * S, C]])


def sector_dipole(L: float, rho: float) -> np.ndarray:
    """
    2x2 horizontal matrix of a sector dipole (bending radius rho [m]).
    The curved reference orbit gives *weak focusing*: x'' + x/rho^2 = 0,
    i.e. a thick 'quadrupole' with k = 1/rho^2.  Vertically a sector dipole
    is just a drift: use ``drift(L)`` for y.

    Parameters
    ----------
    L : float
        Length of dipole [m].
    rho : float
        Bending radius of dipole [m].

    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 2) representing the thick dipole transfer 
        matrix in 2D.
    """
    return thick_quad(1.0 / rho**2, L)


def sector_dipole_dispersion(L: float, rho: float) -> np.ndarray:
    """
    Dispersion column (R16, R26) of a sector dipole.
    A particle with momentum deviation delta follows x'' + x/rho^2 = delta/rho,
    which adds  x += R16*delta,  x' += R26*delta  with
    R16 = rho (1 - cos(L/rho)),  R26 = sin(L/rho).

    Parameters
    ----------
    L : float
        Length of dipole [m].
    rho : float
        Bending radius of dipole [m].

    Returns
    -------
    np.ndarray
        Numpy array of shape (2, 1) representing the dispersion column of the 
        diple transfer matrix.
    """
    theta = L / rho
    return np.array([[rho * (1.0 - np.cos(theta))], [np.sin(theta)]])


def block_4d(Mx: np.ndarray, My: np.ndarray) -> np.ndarray:
    """
    4x4 block-diagonal matrix acting on (x, px, y, py).
    
    Parameters
    ----------
    Mx : np.ndarray
        Numpy array of shape (2, 2) representing horizontal transfer matirx.
    My : np.ndarray
        Numpy array of shape (2, 2) representing vertical transfer matirx.

    Returns
    -------
    np.ndarray
        Numpy array of shape (4, 4) representing the full 4D transverse 
        transfer matrix.
    """
    M = np.zeros((4, 4))
    M[:2, :2] = Mx
    M[2:, 2:] = My
    return M


def matrix_5d(Mx: np.ndarray, My: np.ndarray, d=(0.0, 0.0)) -> np.ndarray:
    """
    5x5 matrix acting on (x, px, y, py, delta) with dispersion column d.
    
    Parameters
    ----------
    Mx : np.ndarray
        Numpy array of shape (2, 2) representing horizontal transfer matirx.
    My : np.ndarray
        Numpy array of shape (2, 2) representing vertical transfer matirx.
    d : np.ndarray | tuple | list
        Numpy array of shape (1, 2) or tuple or list of two elements representing 
        the dispersion column.

    Returns
    -------
    np.ndarray
        Numpy array of shape (5, 5) representing the full 5D transverse 
        transfer matrix.
    """
    M = np.eye(5)
    M[:4, :4] = block_4d(Mx, My)
    M[0, 4], M[1, 4] = d[0], d[1]
    return M


def apply_matrix(particles: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply a matrix to a bunch of N particles represneted by a numpy array of 
    shape (n, N).
    `M` can be one (n, n) matrix, or one matrix per particle with shape
    (N, n, n).  If M is smaller than the number of rows in `particles` (e.g. 
    4x4 on a 5-row particles object) only the first rows are transformed.

    Parameters
    ----------
    particles : np.ndarray
        Numpy array of shape (m, N) representing particles coordinates. m can be 
        2, 4, or 5.
    M : np.ndarray
        Numpy array of shape (N, n, n) or (n, n) representing the transfer matrix. 
        When M is 3 dimensional, it represents particle by particle matrices. n 
        can be less than n

    Returns
    -------
    np.ndarray
        Numpy array of shape (N, m) representing the transformed particles.
    """
    p = np.array(particles, dtype=float, copy=True)
    n = M.shape[-1]
    if M.ndim == 2:
        p = np.matmul(M, p)
    else:
        p[:n] = np.einsum('nij,jn->in', M, p[:n])
    return p


# ---------------------------------------------------------------------------
# Thin multipole kick
# ---------------------------------------------------------------------------
def thin_multipole(particles: np.ndarray, knl=(), ksl=()) -> np.ndarray:
    """
    Apply a thin multipole kick (MAD-X / Xsuite convention).

        Δpx - i Δpy = - Σ_n (knl[n] + i ksl[n]) / n! (x + i y)^n

    Parameters
    ----------
    particles : array (4, N) or (5, N)
        Rows (x, px, y, py[, delta]).  If delta is present the kick is
        divided by (1 + delta) (more rigid particles are deflected less).
    knl : sequence of floats
        Integrated normalised normal multipole strengths, index n = multipole order:
        0 dipole [rad], 1 quadrupole [1/m], 2 sextupole [1/m^2],
        3 octupole [1/m^3].  Missing orders count as zero.
    ksl : sequence of floats
        Integrated normalised skew multipole strengths, index n = multipole order:
        0 dipole [rad], 1 quadrupole [1/m], 2 sextupole [1/m^2],
        3 octupole [1/m^3].  Missing orders count as zero.

    Returns
    -------
    np.ndarray
        New array with px, py updated (x, y are unchanged by a thin kick).
    """
    p = np.array(particles, dtype=float, copy=True)
    knl = np.atleast_1d(np.asarray(knl, dtype=float))
    ksl = np.atleast_1d(np.asarray(ksl, dtype=float))
    n_max = max(len(knl), len(ksl))
    knl = np.pad(knl, (0, n_max - len(knl)))
    ksl = np.pad(ksl, (0, n_max - len(ksl)))

    z = p[0, :] + 1j * p[2, :]          # x + i y
    kick = np.zeros(p.shape[1], dtype=complex)
    zn = np.ones_like(z)                # (x + i y)^n, built up step by step
    for n in range(n_max):              # loop over multipole orders, not particles
        kick += (knl[n] + 1j * ksl[n]) / factorial(n) * zn
        zn = zn * z
    if p.shape[0] > 4:
        kick = kick / (1.0 + p[4, :])
    p[1, :] += -kick.real               # Δpx = -Re(...)
    p[3, :] += kick.imag                # Δpy = +Im(...)
    return p


# ---------------------------------------------------------------------------
# Element objects (used to build lattices and track through them)
# ---------------------------------------------------------------------------
# Every element has a ``length`` and a method ``track(particles)`` returning
# the new particle array.  Linear elements also have ``matrix(delta)``
# returning their 5x5 matrix for momentum deviation ``delta``.
@dataclass
class Drift:
    """
    Field-free drift of length L [m].
    """
    L: float
    name: str = "drift"

    @property
    def length(self) -> float:
        return self.L

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        return matrix_5d(drift(self.L), drift(self.L))

    def track(self, particles):
        return apply_matrix(particles, self.matrix()[:particles.shape[0], :particles.shape[0]])


@dataclass
class Quadrupole:
    """
    Thick quadrupole. k > 0 focuses in x, defocuses in y.
    With a delta column the strength is k/(1 + delta) for each particle
    (this is what gives chromaticity).
    """
    k: float
    L: float
    name: str = "quad"

    @property
    def length(self) -> float:
        return self.L

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        k = self.k / (1.0 + delta)
        return matrix_5d(thick_quad(k, self.L), thick_quad(-k, self.L))

    def track(self, particles):
        if particles.shape[0] < 5:
            return apply_matrix(particles, self.matrix()[:4, :4])
        # one matrix per particle, built vectorised from the formulas of thick_quad
        k = self.k / (1.0 + particles[4, :])
        M = np.zeros((len(particles), 4, 4))
        M[:, :2, :2] = _thick_quad_vec(k, self.L)
        M[:, 2:, 2:] = _thick_quad_vec(-k, self.L)
        return apply_matrix(particles, M)


def _thick_quad_vec(k: np.ndarray, L: float) -> np.ndarray:
    """
    Vectorised thick_quad for an array of strengths k, shape (N, 2, 2).
    
    Parameters
    ----------
    k : np.ndarray
        Numpy array of shape (N, ) of quandrupole strengths [1/m] that have 
        been scaled with delta.
    L : float
        Length of quadrupole [m].

    Returns
    -------
    np.ndarray
        Numpy array of shape (N, 2, 2) of 2x2 matrices.
    """
    k = np.asarray(k, dtype=float)
    sk = np.sqrt(np.abs(k)) + 1e-300
    pos = k > 0
    C = np.where(pos, np.cos(sk * L), np.cosh(sk * L))
    S = np.where(pos, np.sin(sk * L), np.sinh(sk * L)) / sk
    small = np.abs(k) * L**2 < 1e-10
    C = np.where(small, 1.0 - k * L**2 / 2, C)
    S = np.where(small, L - k * L**3 / 6, S)
    kS = np.where(pos, -k*S, k*S)
    M = np.empty(k.shape + (2, 2))
    M[..., 0, 0] = C
    M[..., 0, 1] = S
    M[..., 1, 0] = kS
    M[..., 1, 1] = C
    return M


@dataclass
class SectorBend:
    """
    Sector dipole of length L [m] and bending radius rho [m].
    Linear model: weak focusing 1/rho^2 in x, drift in y, and the
    dispersion column (R16, R26) for off-momentum particles.
    """
    L: float
    rho: float
    name: str = "bend"

    @property
    def length(self) -> float:
        return self.L

    @property
    def angle(self) -> float:
        return self.L / self.rho

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        return matrix_5d(sector_dipole(self.L, self.rho), drift(self.L),
                         sector_dipole_dispersion(self.L, self.rho))

    def track(self, particles):
        n = particles.shape[0]
        return apply_matrix(particles, self.matrix()[:n, :n])


@dataclass
class ThinMultipole:
    """
    Zero-length multipole kick, see :func:`thin_multipole`.
    """
    knl: list = field(default_factory=list)
    ksl: list = field(default_factory=list)
    name: str = "multipole"

    @property
    def length(self) -> float:
        return 0.0

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        """Linear part (quadrupole components only) around x = y = 0."""
        k1 = (self.knl[1] if len(self.knl) > 1 else 0.0) / (1.0 + delta)
        k1s = (self.ksl[1] if len(self.ksl) > 1 else 0.0) / (1.0 + delta)
        M = np.eye(5)
        M[1, 0], M[3, 2] = -k1, k1
        M[1, 2], M[3, 0] = k1s, k1s
        return M

    def track(self, particles):
        return thin_multipole(particles, self.knl, self.ksl)


@dataclass
class LinearMap:
    """
    Arbitrary linear map given by a matrix (4x4 or 5x5).
    ``M`` may also have shape (N, n, n): one matrix per particle (used for
    tune scans where every particle sits in a machine with its own tune).
    """
    M: np.ndarray
    L: float = 0.0
    name: str = "linear_map"

    @property
    def length(self) -> float:
        return self.L

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        if self.M.ndim != 2:
            raise ValueError("matrix() is undefined for per-particle maps")
        if self.M.shape[0] == 5:
            return self.M.copy()
        M = np.eye(5)
        M[:4, :4] = self.M
        return M

    def track(self, particles):
        n = min(self.M.shape[-1], particles.shape[0])
        if self.M.ndim == 2:
            return apply_matrix(particles, self.M[:n, :n])
        return apply_matrix(particles, self.M[:, :n, :n])


@dataclass
class Marker:
    """
    Zero-length marker, does nothing (useful to name a location).
    """
    name: str = "marker"

    @property
    def length(self) -> float:
        return 0.0

    def matrix(self, delta: float = 0.0) -> np.ndarray:
        return np.eye(5)

    def track(self, particles):
        return np.array(particles, dtype=float, copy=True)
