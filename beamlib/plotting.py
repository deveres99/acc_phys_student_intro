"""
Plotting helpers with consistent colours, units and labels.

* Horizontal plane is always black (``COLORS['x']``), vertical always red.
* The code works in metres and radians; plots show mm and mrad
  (factor ``MM = 1e3``).
* Normalised coordinates x_hat = x/sqrt(beta) have units sqrt(m).  We plot
  them multiplied by 1e3, i.e. in sqrt(mm mrad): then x_hat^2 + p_hat^2 = eps_i
  reads directly in mm mrad.   (1 mm mrad = 1e-6 m rad = 1 um.)
* Phase-space plots default to normalised coordinates, where linear motion is
  a circle.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd

import matplotlib.pyplot as plt
import numpy as np

from .optics import normalise

__all__ = ["COLORS", "MM", "phase_space_labels", "phase_space", "ellipse_points",
           "plot_ellipse", "poincare", "resonance_lines", "tune_diagram",
           "amplitude_colors", "fraction_label", "setup"]

COLORS = {"x": "black", "y": "red", "sum": "green", "theory": "blue",
          "accent": "orange", "grey": "0.6"}
MM = 1e3

_LABELS = {
    ("x", False): ("x [mm]", "x' = px [mrad]"),
    ("y", False): ("y [mm]", "y' = py [mrad]"),
    ("x", True): (r"$\hat{x}$ [$\sqrt{\mu\text{m}}$]",
                  r"$\hat{p}_x$ [$\sqrt{\mu\text{m}}$]"),
    ("y", True): (r"$\hat{y}$ [$\sqrt{\mu\text{m}}$]",
                  r"$\hat{p}_y$ [$\sqrt{\mu\text{m}}$]"),
}


def phase_space_labels(ax, plane: str = "x", normalised: bool = True):
    """
    Put the standard axis labels on a phase-space plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to set labels/
    plane : str
        String representing axis, can be 'x' or 'y'.
    normalised : bool
        True if normalised coordinates are plotted, False if physical coordinates
        are plotted.

    Returns
    -------
    """
    xl, yl = _LABELS[(plane, normalised)]
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)


def phase_space(ax, u, pu, beta=None, alpha=None, plane="x", normalised=True,
                s=1.0, color=None, **kw):
    """
    Scatter plot of phase space.
    u, pu : physical coordinates [m], [rad] (any shape, flattened).
    If ``normalised`` (default) and beta/alpha are given, the points are
    converted to Courant-Snyder coordinates and the axes are made square.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    u : np.ndarray
        Numpy array of particle positions in [m].
    pu : np.ndarray
        Numpy array of particle momenta in [rad].
    beta : float
        Twiss beta function in [m].
    alpha : float
        Twiss alpha [1].
    plane : str
        'x' or 'y'.
    normalised : bool
        True if normalised coordinates are plotted, False if physical coordinates
        are plotted.
    s : float
        Marker size on scatter plot.
    color : str
        Colour to plot with.
    **kw

    Returns
    -------
    matplotlib.collections.PathCollection
        To be passed to e.g. colorbar.
    """
    u, pu = np.ravel(u), np.ravel(pu)
    if normalised:
        if beta is None:
            raise ValueError("normalised=True needs beta and alpha")
        u, pu = normalise(u, pu, beta, alpha)
    if color is None and "c" not in kw:
        color = COLORS[plane]
    sc = ax.scatter(u * MM, pu * MM, s=s, color=color, lw=0, **kw)
    phase_space_labels(ax, plane, normalised)
    if normalised:
        ax.set_aspect("equal", adjustable="box")
    return sc


def ellipse_points(eps, beta, alpha, n: int = 400, center=(0.0, 0.0)):
    """
    Points (x, px) on the ellipse  gamma x^2 + 2 alpha x px + beta px^2 = eps.
    
    Parameters
    ----------
    eps : float
        Value of emittance.
    beta : float
        Twiss beta function in [m].
    alpha : float
        Twiss alpha [1].
    n : int
        Number of points.
    center : tuple
        Tuple specifying the centre of the ellipse.

    Returns
    -------
    np.ndarray
        Position coordinates along the ellipse [m].
    np.ndarray
        Angle coordinates along the ellipse [rad].
    """
    t = np.linspace(0, 2 * np.pi, n)
    r = np.sqrt(eps)
    xh, ph = r * np.cos(t), -r * np.sin(t)
    x = np.sqrt(beta) * xh
    px = (ph - alpha * xh) / np.sqrt(beta)
    return x + center[0], px + center[1]


def plot_ellipse(ax, eps, beta, alpha, normalised=False, center=(0.0, 0.0), **kw):
    """
    Draw the phase-space ellipse of single-particle emittance eps [m rad].
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    eps : float
        Value of emittance.
    beta : float
        Twiss beta function in [m].
    alpha : float
        Twiss alpha [1].
    normalised : bool
        True if normalised coordinates are plotted, False if physical coordinates
        are plotted.
    center : tuple
        Tuple specifying the centre of the ellipse.
    **kw

    Returns
    -------
    list[matplotlib.lines.Line2D]
    """
    x, px = ellipse_points(eps, beta, alpha, center=center)
    if normalised:
        x, px = normalise(x, px, beta, alpha)
    kw.setdefault("color", COLORS["theory"])
    kw.setdefault("lw", 1.5)
    return ax.plot(x * MM, px * MM, **kw)


def amplitude_colors(n: int, cmap: str = "viridis"):
    """
    n colours from a colormap, for particles of increasing amplitude.

    Parameters
    ----------
    n : int
        Number of colours to extract from colormap.
    cmap : str
        Name of a matplotlib colormap.
    
    Returns
    -------
    list of colours
    """
    return plt.get_cmap(cmap)(np.linspace(0.05, 0.95, n))


def poincare(ax, u, pu, beta=None, alpha=None, plane="x", normalised=True,
             s=0.6, cmap="viridis", colorbar_label=None, values=None):
    """
    Poincaré section: turn-by-turn points of several particles.
    u, pu : arrays of shape (n_turns, n_particles).  Each particle gets its
    own colour (by index, or by ``values`` if given, e.g. initial amplitude).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    u : np.ndarray
        Numpy array of particle positions in [m], shape (n_turns, n_particles).
    pu : np.ndarray
        Numpy array of particle momenta in [rad], shape (n_turns, n_particles).
    beta : float
        Twiss beta function in [m].
    alpha : float
        Twiss alpha [1].
    plane : str
        Can be 'x' or 'y'.
    normalised : bool
        True if normalised coordinates are plotted, False if physical coordinates
        are plotted.
    s : float
        Marker size on scatter plot.
    cmap : str
        Name of a matplotlib colormap.
    colorbar_label : str
        Label to put on colorbar.
    values : iterable

    Returns
    -------
    matplotlib.collections.PathCollection
        To be passed to e.g. colorbar.
    """
    u, pu = np.asarray(u), np.asarray(pu)
    if normalised:
        u, pu = normalise(u, pu, beta, alpha)
    npart = u.shape[1]
    vals = np.arange(npart) if values is None else np.asarray(values)
    c = np.broadcast_to(vals, u.shape)
    ok = np.isfinite(u) & np.isfinite(pu)
    sc = ax.scatter(u[ok] * MM, pu[ok] * MM, c=c[ok], s=s, lw=0, cmap=cmap)
    phase_space_labels(ax, plane, normalised)
    if normalised:
        ax.set_aspect("equal", adjustable="box")
    if colorbar_label is not None:
        plt.colorbar(sc, ax=ax, label=colorbar_label)
    return sc


def resonance_lines(max_order: int, qx_range=(0, 1), qy_range=(0, 1)):
    """
    List resonances  m Qx + n Qy = p  with |m| + |n| <= max_order.
    Returns a list of (m, n, p, order).  Lines are de-duplicated
    (e.g. 2Qx = 2 is the same line as Qx = 1 and gets order 1).

    Parameters
    ----------
    max_order : int
        Maximum resonance order to get resonance line for.
    qx_range : tuple
        Tuple representing range of horizontal tunes.
    qy_range : tuple
        Tuple representing range of vertical tunes.

    Returns
    -------
    list[tuple[int]]
        Sorted list of unique tuples (m, n, p, order), representing lines of 
        the form m * qx + n * qy = p.
    """
    out = set()
    for order in range(1, max_order + 1):
        for m in range(-order, order + 1):
            n_abs = order - abs(m)
            for n in {n_abs, -n_abs}:
                if m < 0 or (m == 0 and n < 0):
                    continue          # (m, n, p) and (-m, -n, -p) are the same line
                corners = [m * qx + n * qy for qx in qx_range for qy in qy_range]
                for p in range(int(np.floor(min(corners))), int(np.ceil(max(corners))) + 1):
                    g = gcd(gcd(abs(m), abs(n)), abs(p))
                    mm, nn, pp = m // g, n // g, p // g
                    out.add((mm, nn, pp, abs(mm) + abs(nn)))
    return sorted(out, key=lambda r: (r[3], r))


def tune_diagram(ax, max_order: int = 4, qx_range=(0, 1), qy_range=(0, 1),
                 label: bool = True, lw_scale: float = 2.0, alpha: float = 0.8,
                 orders_cmap=None):
    """
    Draw resonance lines m Qx + n Qy = p up to ``max_order``.
    Lower orders are drawn thicker and darker (they are stronger).
    Skew resonances (n odd) are dashed: they need skew magnets (or a
    vertical orbit) to be driven.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis on which to plot.
    max_order : int
        Maximum resonance order to get resonance line for.
    qx_range : tuple
        Tuple representing range of horizontal tunes.
    qy_range : tuple
        Tuple representing range of vertical tunes.
    label : bool
        True if resonances should be labelled, False if not.
    lw_scale : float
        Scale to set line width depending on resonance order.
    alpha : float
        Opacity (1 for fully opaque).
    orders_cmap : dict{int: str}
        Dictionary with integers (corresponding to orders) as keys and strings 
        (corresponding to colours) as values.
    
    Returns
    -------
    matplotlib.axes.Axes
        Axis with resonances plotted.
    """
    colors = orders_cmap or {1: "k", 2: "0.15", 3: "#9467bd", 4: "#ff7f0e",
                             5: "#8c564b", 6: "0.5", 7: "0.65", 8: "0.75"}
    x0, x1 = qx_range
    y0, y1 = qy_range
    seen_orders = set()
    for m, n, p, order in resonance_lines(max_order, qx_range, qy_range):
        ls = "--" if n % 2 else "-"
        lw = lw_scale * (max_order + 1 - order) / max_order
        kw = dict(color=colors.get(order, "0.7"), lw=lw, ls=ls, alpha=alpha, zorder=1)
        if order not in seen_orders and label:
            kw["label"] = f"order {order}"
            seen_orders.add(order)
        if n == 0:
            q = p / m
            if x0 <= q <= x1:
                ax.plot([q, q], [y0, y1], **kw)
        else:
            qx = np.linspace(x0, x1, 200)
            qy = (p - m * qx) / n
            ok = (qy >= y0) & (qy <= y1)
            if ok.any():
                ax.plot(qx[ok], qy[ok], **kw)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel(r"$Q_x$")
    ax.set_ylabel(r"$Q_y$")
    ax.set_aspect("equal")
    return ax


def fraction_label(q: float, max_den: int = 8) -> str:
    """Nice label like '1/3' for a tune value."""
    return str(Fraction(q).limit_denominator(max_den))


def setup(dpi: int = 100):
    """Consistent matplotlib defaults for all notebooks."""
    plt.rcParams.update({
        "figure.dpi": dpi,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    })
