"""
beamlib - a small, readable toolkit for transverse beam dynamics exercises.

Modules
-------
elements     transfer matrices and thin multipole kicks
optics       Twiss parameters, normalised coordinates, closed orbit
tracking     turn-by-turn tracking
lattices     ready-made machines (Hénon map, Hénon-like ring, FODO ring)
diagnostics  invariants, tunes, emittances, chaos indicator
plotting     consistent plotting helpers
beams        Gaussian bunches (matched or from any Sigma matrix)
widgets      sliders that fall back to static plots without ipywidgets

Conventions (see each module docstring for details): coordinates
(x, px, y, py, delta) with px = x' (small angles), SI units inside the code
(m, rad), bunches are arrays of shape (N, 4) or (N, 5), multipole kicks
follow MAD-X/Xsuite: Δpx - iΔpy = -Σ (knl_n + i ksl_n)/n! (x + iy)^n.
"""
from . import beams, diagnostics, elements, lattices, optics, plotting, tracking, widgets  # noqa: F401

__version__ = "1.0"
