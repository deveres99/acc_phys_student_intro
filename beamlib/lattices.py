"""
Ready-made machines used in the exercises.

1. ``henon_map_2d``  - the classic 2D quadratic Hénon map (warm-up).
2. ``HenonRing``     - a "Hénon-like" ring: a linear rotation with chosen
   tunes and Twiss parameters, followed by one thin multipole kick.  The
   observation point is just after the rotation, i.e. at the kick entrance.
   All resonance exercises use it.
3. ``fodo_ring``     - a small ring of thick FODO cells with sector dipoles
   (real dispersion, natural chromaticity) and two thin sextupoles, SF and
   SD.  Used for the dispersion and chromaticity exercises.
"""
from __future__ import annotations

import numpy as np

from .elements import Drift, LinearMap, Marker, Quadrupole, SectorBend, ThinMultipole
from .optics import (denormalise, matrix_from_twiss, normalise, one_turn_matrix,
                     twiss_from_matrix)
from .tracking import make_one_turn_map

__all__ = ["henon_map_2d", "HenonRing", "fodo_cell", "fodo_ring", "FODO_DEFAULTS"]


# ---------------------------------------------------------------------------
# 1. Classic Hénon map
# ---------------------------------------------------------------------------
def henon_map_2d(x, p, Q: float):
    """
    One iteration of the classic area-preserving quadratic Hénon map.
    Kick  p -> p + x^2,  then rotate clockwise by mu = 2 pi Q:
        x' =  x cos(mu) + (p + x^2) sin(mu)
        p' = -x sin(mu) + (p + x^2) cos(mu)
    This is exactly a ``HenonRing`` with beta = 1 m, alpha = 0 and a thin
    sextupole of strength knl[2] = -2 (in normalised units), seen *after*
    the kick instead of before it.

    Parameters
    ----------
    x : np.ndarray
        Numpy array of position coordinates.
    p : np.ndarray
        Numpy array of angle coordinates.
    Q : float
        Tune [2pi].

    Returns
    -------
    np.ndarray
        Numpy array of position coordinates after one turn.
    np.ndarray
        Numpy array of angle coordinates after one turn.
    """
    mu = 2 * np.pi * Q
    pk = p + x**2
    return x * np.cos(mu) + pk * np.sin(mu), -x * np.sin(mu) + pk * np.cos(mu)


# ---------------------------------------------------------------------------
# 2. Hénon-like ring
# ---------------------------------------------------------------------------
class HenonRing:
    """Linear one-turn rotation followed by a single thin multipole kick.

    Parameters
    ----------
    Qx, Qy : float or array
        Tunes of the linear part. Arrays (one value per particle) are allowed
        for tune scans: then particle i sees tunes (Qx[i], Qy[i]).
    betx, alfx, bety, alfy : float
        Twiss parameters at the kick [m], [-].
    knl, ksl : sequences
        Thin multipole strengths at the kick (MAD-X/Xsuite convention, see
        :func:`beamlib.elements.thin_multipole`).

    Useful members: ``elements`` (list), ``one_turn_map`` (callable),
    ``normalise(bunch)`` / ``denormalise(bunch)``.
    """

    def __init__(self, Qx=0.31, Qy=0.32, betx=4.0, alfx=0.0, bety=4.0, alfy=0.0,
                 knl=(), ksl=()):
        self.Qx, self.Qy = Qx, Qy
        self.betx, self.alfx, self.bety, self.alfy = betx, alfx, bety, alfy
        self.knl, self.ksl = list(knl), list(ksl)
        Mx = _rotation_matrices(Qx, betx, alfx)
        My = _rotation_matrices(Qy, bety, alfy)
        if Mx.ndim == 2 and My.ndim == 2:
            M = np.zeros((4, 4))
            M[:2, :2], M[2:, 2:] = Mx, My
        else:
            n = max(Mx.shape[0] if Mx.ndim == 3 else 1, My.shape[0] if My.ndim == 3 else 1)
            M = np.zeros((n, 4, 4))
            M[:, :2, :2], M[:, 2:, 2:] = Mx, My
        self.elements = [LinearMap(M, name="rotation"),
                         ThinMultipole(self.knl, self.ksl, name="kick")]
        self.one_turn_map = make_one_turn_map(self.elements)

    def normalise(self, bunch):
        """Physical (4, N) bunch -> normalised (x_hat, p_hat_x, y_hat, p_hat_y)."""
        b = np.asarray(bunch, dtype=float)
        out = np.empty_like(b[..., :4])
        out[0, ...], out[1, ...] = normalise(b[0, ...], b[1, ...], self.betx, self.alfx)
        out[2, ...], out[3, ...] = normalise(b[2, ...], b[3, ...], self.bety, self.alfy)
        return out

    def denormalise(self, bunch_n):
        """Normalised (4, N) bunch -> physical (x, px, y, py)."""
        b = np.asarray(bunch_n, dtype=float)
        out = np.empty_like(b[..., :4])
        out[0, ...], out[1, ...] = denormalise(b[0, ...], b[1, ...], self.betx, self.alfx)
        out[2, ...], out[3, ...] = denormalise(b[2, ...], b[3, ...], self.bety, self.alfy)
        return out

    def __repr__(self):
        return (f"HenonRing(Qx={self.Qx}, Qy={self.Qy}, betx={self.betx}, "
                f"alfx={self.alfx}, bety={self.bety}, alfy={self.alfy}, "
                f"knl={self.knl}, ksl={self.ksl})")


def _rotation_matrices(Q, beta, alpha):
    """matrix_from_twiss for scalar or array Q; shape (2, 2) or (N, 2, 2)."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 0:
        return matrix_from_twiss(beta, alpha, 2 * np.pi * float(Q))
    mu = 2 * np.pi * Q
    gamma = (1 + alpha**2) / beta
    c, s = np.cos(mu), np.sin(mu)
    M = np.empty(Q.shape + (2, 2))
    M[..., 0, 0] = c + alpha * s
    M[..., 0, 1] = beta * s
    M[..., 1, 0] = -gamma * s
    M[..., 1, 1] = c - alpha * s
    return M


# ---------------------------------------------------------------------------
# 3. FODO ring with dispersion and two sextupoles
# ---------------------------------------------------------------------------
FODO_DEFAULTS = dict(
    n_cells=8,        # number of FODO cells (ring circumference = n_cells * L_cell)
    L_cell=6.0,       # cell length [m]
    L_quad=0.4,       # length of one full quadrupole [m] (QF and QD)
    L_bend=1.6,       # length of one sector dipole [m] (2 per cell)
    kf=None,          # QF strength [1/m^2]; None -> default value below
    kd=None,          # QD strength [1/m^2] (positive number; QD gets -kd)
)


def fodo_cell(kf, kd, L_cell=6.0, L_quad=0.4, L_bend=1.6, n_cells=8,
              sf=None, sd=None, tag=""):
    """One FODO cell starting and ending in the middle of QF.

    Layout:  QF/2 - d - B - d - QD/2 [SD] QD/2 - d - B - d - QF/2 [SF]

    ``sf``/``sd`` are integrated sextupole strengths knl[2] [1/m^2] placed
    as thin kicks at the centre of QD and at the end of the cell (= centre of
    the next QF).  ``None`` means no sextupole in this cell.
    """
    theta = 2 * np.pi / (2 * n_cells)          # bending angle per dipole
    rho = L_bend / theta
    L_drift = (L_cell - 2 * L_quad - 2 * L_bend) / 4   # QF/2 + QD + QF/2 = 2 L_quad
    if L_drift < 0:
        raise ValueError("Elements do not fit in the cell")
    els = [
        Quadrupole(kf, L_quad / 2, name=f"qf{tag}_b"),
        Drift(L_drift), SectorBend(L_bend, rho, name=f"b{tag}_1"), Drift(L_drift),
        Quadrupole(-kd, L_quad / 2, name=f"qd{tag}_a"),
        ThinMultipole([0, 0, sd], name="sd") if sd is not None else Marker(f"qd{tag}_centre"),
        Quadrupole(-kd, L_quad / 2, name=f"qd{tag}_b"),
        Drift(L_drift), SectorBend(L_bend, rho, name=f"b{tag}_2"), Drift(L_drift),
        Quadrupole(kf, L_quad / 2, name=f"qf{tag}_a"),
        ThinMultipole([0, 0, sf], name="sf") if sf is not None else Marker(f"qf{tag}_centre"),
    ]
    return els


def fodo_ring(kf: float = 1.02, kd: float = 1.08, k2_sf: float = 0.0,
              k2_sd: float = 0.0, n_cells: int = 8, L_cell: float = 6.0,
              L_quad: float = 0.4, L_bend: float = 1.6):
    """Ring of ``n_cells`` FODO cells with one SF and one SD sextupole.

    The ring starts (and ends) at the centre of a QF, where alpha_x = alpha_y
    = 0, D' = 0 and beta_x, D are maximal.  SF (strength ``k2_sf`` [1/m^2])
    sits at the end of the *last* cell (centre of QF, i.e. the start/end of the
    ring) and SD (``k2_sd``) at the centre of the QD of the *first* cell.
    Both are at places with D > 0, but with very different beta_x/beta_y,
    so they can correct the two chromaticities independently.
    """
    els = []
    for i in range(n_cells):
        els += fodo_cell(kf, kd, L_cell, L_quad, L_bend, n_cells,
                         sf=k2_sf if i == n_cells - 1 else None,
                         sd=k2_sd if i == 0 else None, tag=str(i))
    return els
