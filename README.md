# FFT-FD Method

A solver for the pressure equation of variable-density flows in a box that is periodic in the horizontal directions,

$$\rho\,\nabla\cdot\left(\frac{1}{\rho}\nabla p\right) = \nabla^2 p - \frac{\nabla\rho}{\rho}\cdot\nabla p = f,
\qquad \frac{\partial p}{\partial z} = g \ \text{at the bottom}, \qquad p = p_{\text{top}} \ \text{at the top}.$$

It combines a 2D Fast Fourier Transform in $x$ and $y$ with second-order finite differences on a uniform or stretched grid in $z$ (FFT-FD), which reduces the problem to independent tridiagonal systems, one per Fourier mode.

- **Constant density** (Poisson equation): direct solve.
- **Density that depends on $z$ only**: direct solve, at the same cost.
- **General density $\rho(x,y,z)$**: fixed-point (Picard) or GMRES iteration in which every step is one direct solve, optionally with the horizontally averaged stratification included in the direct solver.

This code is used in:
* **A Mixed 2D Discrete Fourier Transform and Finite Difference Method for Solving a 3D Poisson Equation**, presented at the *XXXVI Jornada de Matemática de la Zona Sur*, Universidad Católica de Temuco (Campus San Juan Pablo II), April 24–26, 2024 ([abstract](https://drive.google.com/file/d/1lkNVFLabjullLxWOcN3T7dGjf0716NA7/view), [slides](https://drive.google.com/file/d/1h-TjD_LDXyfRLDOTw9K4RFtJ1Dl4CkRK/view)).
* The numerical implementation of a coupled atmosphere-wildfire model ([`dsanmartin/wildfire`](https://github.com/dsanmartin/wildfire); GPU version in [`dsanmartin/wildfires`](https://github.com/dsanmartin/wildfires)), described in [San Martin & Torres (2025)](https://doi.org/10.1016/j.cpc.2025.109746) and the PhD thesis ([doi:10.71959/k76w-qn05](https://doi.org/10.71959/k76w-qn05)).

The original implementation of the solver, developed for the thesis, is kept in `fftfd_original.py` (the same algorithm is in [`src/pressure.py`](https://github.com/dsanmartin/wildfire/blob/main/src/pressure.py) of the coupled-model repository). `solver.py` is a batched, multithreaded rewrite of it, extended to stretched vertical grids and variable density.

## Installation

Python 3.11 with NumPy, SciPy, Numba and Matplotlib:

```bash
pip install -r requirements.txt
```

Check that everything works (a few seconds, writes nothing):

```bash
python smoke_test.py
```

## Usage

```python
import numpy as np
from solver import FFTFDSolver, stretched_grid, mean_stratification

# x and y include both end points (the last node is the periodic image of the first)
x = np.linspace(0, 2 * np.pi, 65)
y = np.linspace(0, 2 * np.pi, 65)
z = stretched_grid(0, 2 * np.pi, 65, k=2.0)   # k = 0 gives a uniform grid

solver = FFTFDSolver(x, y, z)                  # bottom="one-sided" (default) or "ghost"

# f and rho are arrays of shape (Nx, Ny, Nz)
p = solver.solve(f)                            # constant density
p = solver.solve(f, p_top=p_top, dpdz_bottom=g)   # non-zero boundary data, shape (Nx, Ny)

# general density: fixed point or GMRES
p, info = solver.solve_variable_density(f, rho, method="picard", tol=1e-8)
p, info = solver.solve_variable_density(f, rho, method="gmres", tol=1e-8)
print(info["iterations"], info["converged"])

# include the mean vertical stratification in the direct solver
strat = FFTFDSolver(x, y, z, beta=mean_stratification(rho, x, y, z))
p, info = strat.solve_variable_density(f, rho, method="gmres")
```

Practical notes:

- **Transform length.** The FFT length is the number of nodes minus one in each horizontal direction. Use `Nx = Ny = 2^n + 1` (for example 65, 129, 257): with 128 nodes the length is 127, a prime, and the solve is about three times slower.
- **Threads.** The tridiagonal solves run on Numba's thread pool (`solver.set_threads(n)`) and the FFTs on SciPy's (`FFTFDSolver(..., workers=n)`).
- **Which iteration.** The fixed-point iteration is the cheapest per step and converged for every localized density variation we tried (density ratios up to 101), but it slows down and diverges when the domain spans many density scale heights. GMRES converges in those cases too, and the stratified operator makes the iteration count independent of the background stratification.

## Repository layout

| Path | Content |
|---|---|
| `solver.py` | The solver (`FFTFDSolver`), finite-difference weights, stretched grid. |
| `manufactured.py` | Manufactured solutions with analytical source terms. |
| `fftfd_original.py` | Verbatim copy of the original implementation: uniform and non-uniform solvers and the fixed-point iteration for variable density. Timing baseline of the paper and reference for `smoke_test.py`. |
| `smoke_test.py` | Quick check of every solver path, and of `solver.py` against the original implementation; stores nothing. |
| `experiments.py` | Numerical experiments of the paper; writes `paper/data/results.json`. |
| `fds_replay.py` | Replays FDS density and pressure fields through the solver; writes `paper/data/fds_replay.json`. |
| `figures.py`, `tables.py` | Figures and LaTeX tables of the paper, from the data files. |
| `paper/` | Manuscript sources, data, figures and tables. |
| `fds/` | FDS input files for the fire case (see `fds/README.md`). |

## Reproducing the paper

```bash
python experiments.py     # manufactured problems and timings (about 4 minutes)
python fds_replay.py fds  # fire replay; needs the FDS output, see fds/README.md
python figures.py         # paper/figures/  (needs a LaTeX installation for the labels)
python tables.py          # paper/tables/, including the timing numbers quoted in the text
cd paper && latexmk -pdf main.tex
```

`experiments.py` accepts group names to run only part of the suite: `const`, `nonuniform`, `vd`, `perf`.
The data files in `paper/data/` are included, so the figures, tables and manuscript can be rebuilt without re-running the experiments or FDS. Timings depend on the machine; those in the paper were measured on an Apple M4.

## License

See `LICENSE`.
