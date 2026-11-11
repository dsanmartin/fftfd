# Legacy code

Earlier versions of the solver and the exploratory notebooks, kept for reference. The current code
is in the repository root (`solver.py`).

| File | Content |
|---|---|
| `fftfd.py` | Original solver: Python loop over the Fourier modes with a Numba Thomas routine. Uniform (`fftfd`) and non-uniform (`fftfd_non_uniform_z`) vertical grids. Still used by `experiments.py` as the timing baseline. |
| `fftfd_parallell.py` | Variant that distributes the modes with `multiprocessing`. |
| `analysis*.ipynb`, `grid.ipynb`, `condition.ipynb` | Notebooks behind the results of the thesis and the first tests of the non-uniform grid, the variable density and the parallel version. |
| `condition_number_*.pdf` | Condition-number plots produced by `condition.ipynb`. |
| `src/`, `include/`, `Makefile` | Unfinished C/OpenMP test. Not functional. |

Run the notebooks from inside this folder so that `from fftfd import *` resolves. They need
`ipykernel` and `ipywidgets` in addition to `requirements.txt`.

Known issues, left as they were:

- `analysis_nonuniform.ipynb`: the stretching parameter `k` is overwritten by a slice index before
  the convergence loop, so the stored errors of the non-uniform solver are wrong (the solver is fine).
- `fftfd_density` / `fftfd_density_a` (direct solver for a general density): not valid when the
  density depends on `x` or `y`, because the density term couples the Fourier modes.
- The solvers assume the domain starts at 0 (`x_max = x[-1]`); `condition.ipynb` uses a domain that does not.
