"""
Smoke test: runs every solver path on small grids and checks the results. Writes nothing.

    python smoke_test.py

No results, figures or tables are produced. Python bytecode is not written and the Numba
compilation cache goes to a temporary folder that is deleted at exit, so the repository is
left untouched. Takes a few seconds, most of it compilation.
"""
import os
import sys
import tempfile

sys.dont_write_bytecode = True
_tmp = tempfile.TemporaryDirectory()
os.environ["NUMBA_CACHE_DIR"] = _tmp.name  # must be set before numba is imported

import numpy as np

import manufactured as mf
from solver import FFTFDSolver, mean_stratification, stretched_grid

L = 2 * np.pi
CHECKS = []


def check(name, ok, detail):
    CHECKS.append(ok)
    print("%-4s %-52s %s" % ("ok" if ok else "FAIL", name, detail))


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    gauss = mf.Gaussian2D(np.pi, np.pi, 0.5, 0.5)
    bump = mf.VonMises(L, L, a=2.0)

    # 1. Constant density: second-order convergence, both bottom closures
    for closure in ["one-sided", "ghost"]:
        err = []
        for N in [17, 33]:
            x = np.linspace(0, L, N)
            P, _, F = mf.fields(x, x, x, mf.z_thesis, gauss)
            err.append(rel(FFTFDSolver(x, x, x, bottom=closure).solve(F), P))
        order = np.log2(err[0] / err[1])
        check("constant density, %s closure" % closure, 1.7 < order < 2.5 and err[1] < 2e-2, "error %.1e, order %.2f" % (err[1], order))

    # 2. Stretched vertical grid resolves a bottom layer better than the uniform one
    x = np.linspace(0, L, 17)
    errs = {}
    for k in [0.0, 4.0]:
        z = stretched_grid(0.0, L, 33, k)
        P, _, F = mf.fields(x, x, z, lambda Z: mf.z_layer(Z, L, 0.25), bump)
        errs[k] = rel(FFTFDSolver(x, x, z).solve(F), P)
    check("stretched grid, bottom-layer solution", errs[4.0] < 0.05 and errs[4.0] < 0.1 * errs[0.0], "uniform %.1e, stretched %.1e" % (errs[0.0], errs[4.0]))

    # 3. Non-zero boundary data: the solver must reproduce a field built from its own operator
    x = np.linspace(0, L, 17)
    z = stretched_grid(0.0, L, 17, 1.0)
    s = FFTFDSolver(x, x, z)
    X, Y, Z = np.meshgrid(x, x, z, indexing="ij")
    q = np.cos(X) * np.sin(2 * Y) * (1 + Z + 0.3 * Z**2)  # dq/dz != 0 at the bottom, q != 0 at the top
    f = -5 * q + np.cos(X) * np.sin(2 * Y) * 0.6  # exact Laplacian (the z part is quadratic: no FD error)
    g = np.cos(X[:, :, 0]) * np.sin(2 * Y[:, :, 0])
    check("non-zero top value and bottom gradient", rel(s.solve(f, p_top=q[:, :, -1], dpdz_bottom=g), q) < 1e-10,
          "error %.1e" % rel(s.solve(f, p_top=q[:, :, -1], dpdz_bottom=g), q))

    # 4. Variable density: fixed point and GMRES converge to the same second-order solution
    x = np.linspace(0, L, 33)
    P, RHO, F = mf.fields(x, x, x, mf.z_thesis, bump, theta=4.0, Zr=lambda Z: mf.z_layer(Z, L, 0.75), Hr=bump)
    s = FFTFDSolver(x, x, x)
    pp, ip = s.solve_variable_density(F, RHO, method="picard", tol=1e-10)
    pg, ig = s.solve_variable_density(F, RHO, method="gmres", tol=1e-10)
    check("variable density, fixed point", ip["converged"] and rel(pp, P) < 2e-2, "%d iterations, error %.1e" % (ip["iterations"], rel(pp, P)))
    check("variable density, GMRES", ig["converged"] and rel(pg, pp) < 1e-7 and ig["iterations"] < ip["iterations"],
          "%d iterations, difference to fixed point %.1e" % (ig["iterations"], rel(pg, pp)))

    # 5. Stratified operator: a density that depends on z only is solved without iterating
    P, RHO, F = mf.fields(x, x, x, mf.z_thesis, bump, strat=10.0 / L)
    sb = FFTFDSolver(x, x, x, beta=mean_stratification(RHO, x, x, x))
    ps, i_s = sb.solve_variable_density(F, RHO, method="picard", tol=1e-10)
    check("stratified operator, density rho(z)", i_s["converged"] and i_s["iterations"] <= 2 and rel(ps, P) < 2e-2,
          "%d iterations, error %.1e" % (i_s["iterations"], rel(ps, P)))

    print("\n%d of %d checks passed" % (sum(CHECKS), len(CHECKS)))
    return 0 if all(CHECKS) else 1


if __name__ == "__main__":
    code = main()
    _tmp.cleanup()
    sys.exit(code)
