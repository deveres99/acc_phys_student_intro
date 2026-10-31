# acc_phys_student_intro
Short python tutorials for students to get started with coding and accelerator physics simulations.

## python_intro
Notebooks to introduce basic python concepts needed to run accelerator physics 
simulations.

`000_python_fundamentals.ipynb` introduces concepts like
- Variables and arithmetic
- `if` / `elif` / `else`
- `for` loops
- Writing functions
- Using loops and functions together

`001_numpy_pandas_matplotlib.ipynb` introduces the basics of some core 
scientific computing packages, such as numpy, pandas and matplotlib for 
visualising data.

## Transverse beam dynamics with Python
A hands-on course for interns (high-school and first-year university students) who have just had
lectures on transverse beam dynamics. Everything is plain NumPy/SciPy/Matplotlib, no beam-dynamics
library; conventions follow MAD-X/Xsuite so that students can move on to Xsuite afterwards.
The central aim is **intuition for core transverse beam dynamics concepts**.

### Repository layout

```
beamlib/                                  shared toolkit (students are encouraged to read it)
  elements.py                             transfer matrices, thin multipole kick, element classes
  optics.py                               Twiss, normalisation, closed orbit, dispersion, fixed points
  tracking.py                             track(), one-turn-map composer, loss detection
  lattices.py                             Hénon map, Hénon-like ring, FODO ring with SF/SD
  diagnostics.py                          invariants, action-angle, tunes (angle/FFT/NAFF), emittances, chaos indicator
  plotting.py                             consistent phase-space plots, ellipses, Poincaré sections, tune diagrams
  beams.py                                Gaussian bunches from a Sigma matrix
  widgets.py                              sliders with a static fallback when ipywidgets is missing
transverse_beam_dynamics/_src/            ONE source file per exercise (*.nbsrc): edit these (currently not committed)
notebooks/student/                        generated: stubs + self-check asserts
transverse_beam_dynamics/solutions/       generated: full solutions + answers to reflection questions (currently not committed)
tools/build_transverse_notebooks.py       builds student/ and solutions/ from _src/
tools/run_transverse_notebooks.py         executes notebooks in a fresh kernel and checks them
requirements.txt
```

### Quick start

```bash
pip install -r requirements.txt                 # Python >= 3.10
python -m pytest tests                          # or: python tests/run_tests.py
python tools/build_transverse_notebooks.py                 # regenerate notebooks after editing _src/
python tools/run_transverse_notebooks.py notebooks/solutions/*.ipynb            # all must pass
python tools/run_transverse_notebooks.py --student notebooks/student/*.ipynb    # must stop at a stub
```

Notebooks find `beamlib` on their own (the first cell walks up to the repository root), so students only
need to open `transverse_beam_dynamics/student/exNN_*.ipynb` in Jupyter. **Do not edit the generated notebooks**: edit the
`.nbsrc` file and rebuild. In the source, `### BEGIN SOLUTION` / `### END SOLUTION` blocks become
`raise NotImplementedError` in the student version, `%% md solution` cells appear only in the solutions.