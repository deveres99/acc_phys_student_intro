"""
Turn-by-turn tracking.

A *one-turn map* is any function that takes a bunch (array of shape (N, 4)
or (N, 5)) and returns the bunch after one turn.  ``make_one_turn_map``
builds one from a list of elements applied in order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["make_one_turn_map", "track", "TrackResult"]


def make_one_turn_map(elements):
    """
    Compose a list of elements (applied first to last) into one function.
    
    Parameters
    ----------
    elements : list
        List of beamlib elemets.

    Returns
    -------
    callable
        Function to track through list of elements in one step.
    """
    elements = list(elements)

    def one_turn_map(particles):
        p = particles
        for el in elements:
            p = el.track(p)
        return p

    return one_turn_map


@dataclass
class TrackResult:
    """Output of :func:`track`.

    Attributes
    ----------
    coords : array (n_turns + 1, nrow, N) if store=True, else (2, nrow, N)
        Turn-by-turn coordinates; ``coords[0]`` is the initial bunch.
        Lost particles are NaN from the turn they were lost onwards.
    lost_turn : array (N,)
        Turn at which each particle was lost, or n_turns if it survived.
    final : array (nrow, N)
        Coordinates after the last turn (NaN for lost particles).
    """
    coords: np.ndarray
    lost_turn: np.ndarray
    final: np.ndarray
    n_turns: int

    # handy shortcuts: arrays of shape (n_turns + 1, N)
    @property
    def x(self):
        return self.coords[:, 0, :]

    @property
    def px(self):
        return self.coords[:, 1, :]

    @property
    def y(self):
        return self.coords[:, 2, :]

    @property
    def py(self):
        return self.coords[:, 3, :]

    @property
    def survived(self):
        """Boolean array: True for particles that survived all turns."""
        return self.lost_turn == self.n_turns


def track(particles, one_turn_map, n_turns: int, store: bool = True,
          aperture: float = 1.0) -> TrackResult:
    """Track a bunch for ``n_turns`` turns.

    Parameters
    ----------
    particles : np.ndarray
        Numpy array of shape (4, N) or (5, N) representing particle coordinates.
    one_turn_map : callable
        Function to perform bunch -> bunch mapping after one revolution.
    n_turns : int
        Number of turns.
    store : bool
        Keep every turn (memory ~ n_turns * N * ncol * 8 bytes).  With
        ``store=False`` only the initial and final coordinates are kept
        (``coords`` then has two entries: first and last turn).
    aperture : float
        A particle with |x| or |y| > aperture [m] is lost; it is then frozen
        as NaN.  Default 1 m, far outside anything physical in our toy rings,
        but it stops runaway particles from producing huge numbers.

    Returns
    -------
    TrackResult
        Result of tracking containing (turn-by-turn) particle coordinates.
    """
    p = np.array(particles, dtype=float, copy=True)
    if p.ndim == 1:
        p = p[:, None]
    N = p.shape[1]
    lost_turn = np.full(N, n_turns)
    alive = np.ones(N, dtype=bool)
    coords = np.empty((n_turns + 1 if store else 1,) + p.shape)
    coords[0] = p
    with np.errstate(over="ignore", invalid="ignore"):
        for turn in range(1, n_turns + 1):
            # The whole bunch goes through the map (lost particles are NaN and
            # stay NaN). This keeps the particle order intact, which matters
            # for maps with one matrix per particle.
            p = one_turn_map(p)
            bad = alive & ~(np.isfinite(p[:4, :]).all(axis=0)
                            & (np.abs(p[0, :]) <= aperture)
                            & (np.abs(p[2, :]) <= aperture))
            if bad.any():
                lost_turn[bad] = turn
                alive &= ~bad
                p[:4, bad] = np.nan
            if store:
                coords[turn] = p
            if not alive.any():
                if store:
                    coords[turn + 1:] = np.nan
                break
    if not store:
        coords = np.stack([coords[0], p])
    return TrackResult(coords=coords, lost_turn=lost_turn, final=p, n_turns=n_turns)
