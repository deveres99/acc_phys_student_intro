"""
Interactive sliders that degrade gracefully.

``interactive_plot(func, **sliders)`` shows ipywidgets sliders when
ipywidgets is installed (and we are in a notebook); otherwise it simply calls
``func`` once with the default value of every slider, so the static plot
still appears.

Each slider is given as a tuple ``(min, max, step, default)``.
"""
from __future__ import annotations

__all__ = ["HAVE_WIDGETS", "interactive_plot"]

try:  # ipywidgets is optional
    import ipywidgets as _w
    HAVE_WIDGETS = True
except Exception:  # pragma: no cover - depends on the installation
    _w = None
    HAVE_WIDGETS = False


def interactive_plot(func, continuous_update: bool = False, **sliders):
    """Run ``func(**values)`` with sliders, or once with defaults as a fallback.

    Example
    -------
    >>> interactive_plot(lambda Q, k: print(Q, k),
    ...                  Q=(0.0, 0.5, 0.01, 0.31), k=(0, 10, 0.5, 2.0))
    """
    defaults = {name: spec[3] for name, spec in sliders.items()}
    if not HAVE_WIDGETS:
        print("(ipywidgets not available: showing the plot for the default values "
              + ", ".join(f"{k}={v}" for k, v in defaults.items()) + ")")
        return func(**defaults)
    controls = {}
    for name, (lo, hi, step, default) in sliders.items():
        controls[name] = _w.FloatSlider(min=lo, max=hi, step=step, value=default,
                                        description=name,
                                        continuous_update=continuous_update,
                                        readout_format=".4g")
    return _w.interact(func, **controls)
