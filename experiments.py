"""
Numerical experiments for the paper. Writes paper/data/results.json.

    python experiments.py            # everything
    python experiments.py vd perf    # only the selected groups

Groups: const, nonuniform, vd, divform, perf.
"""
import json
import os
import platform
import sys
import time

import numba
import numpy as np
from scipy.linalg import solve_banded

import manufactured as mf
from solver import FFTFDSolver, _solve_modes, log_density_gradient, mean_stratification, set_threads, stretched_grid

L = 2 * np.pi
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "data", "results.json")
MAX_THREADS = numba.config.NUMBA_NUM_THREADS


def rel2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def relinf(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def grid(N, k=0.0, Nz=None):
    x = np.linspace(0, L, N)
    return x, x.copy(), stretched_grid(0.0, L, N if Nz is None else Nz, k)


def rate(r):
    """Asymptotic reduction factor per iteration, from the tail of a residual history."""
    r = np.asarray(r)
    r = r[np.isfinite(r) & (r > 1e-13)]
    if len(r) < 4:
        return float("nan")
    m = min(8, len(r) - 2)
    return float((r[-1] / r[-1 - m]) ** (1.0 / m))


# ---------------------------------------------------------------------------------------
def exp_const():
    """Constant density, uniform grid: the thesis problem. Order of accuracy and Thomas stability."""
    out = {"N": [], "dz": [], "one-sided": {"l2": [], "linf": []}, "ghost": {"l2": [], "linf": []}}
    H = mf.Gaussian2D(np.pi, np.pi, 0.5, 0.5)
    for N in [16, 32, 64, 128, 256, 512]:
        x, y, z = grid(N)
        P, _, F = mf.fields(x, y, z, mf.z_thesis, H)
        out["N"].append(N)
        out["dz"].append(float(z[1] - z[0]))
        for closure in ["one-sided", "ghost"]:
            p = FFTFDSolver(x, y, z, bottom=closure).solve(F)
            out[closure]["l2"].append(rel2(p, P))
            out[closure]["linf"].append(relinf(p, P))
        print("const N=%d  l2=%.3e (one-sided)  %.3e (ghost)" % (N, out["one-sided"]["l2"][-1], out["ghost"]["l2"][-1]), flush=True)

    # Thomas algorithm without pivoting vs LAPACK banded LU with partial pivoting, all modes,
    # for three cell aspect ratios dz/dx and both bottom closures.
    stab = []
    rng = np.random.default_rng(0)
    for name, Lz, k in [("dz = dx", L, 0.0), ("dz = dx / 16", L / 16, 0.0), ("dz = 16 dx", 16 * L, 0.0), ("stretched, k = 4", L, 4.0)]:
        N = 65
        x = np.linspace(0, L, N)
        z = stretched_grid(0.0, Lz, N, k)
        for closure in ["one-sided", "ghost"]:
            s = FFTFDSolver(x, x, z, bottom=closure)
            n, M = s.n, s.kappa.size
            rhs = rng.standard_normal((M, n)) + 1j * rng.standard_normal((M, n))
            sol = rhs.copy()
            _solve_modes(s.lower, s.diag, s.upper, s.kappa, *s.r0, sol)
            worst, dominant, growth = 0.0, 0, 0.0
            for m in range(M):
                kap = s.kappa[m]
                d = s.diag - kap
                up = s.upper.copy()
                d[0] = s.r0[0] + s.r0[1] * kap
                up[0] = s.r0[2] + s.r0[3] * kap
                ab = np.zeros((3, n))
                ab[0, 1:], ab[1], ab[2, :-1] = up[:-1], d, s.lower[1:]
                ref = solve_banded((1, 1), ab, rhs[m])
                worst = max(worst, np.abs(ref - sol[m]).max() / np.abs(ref).max())
                off = np.abs(up).copy()
                off[-1] = 0.0
                off[1:] += np.abs(s.lower[1:])
                dominant += int(np.all(np.abs(d) >= off))
                # element growth of the LU factors without pivoting
                piv = np.empty(n)
                piv[0] = d[0]
                for i in range(1, n):
                    piv[i] = d[i] - s.lower[i] * up[i - 1] / piv[i - 1]
                growth = max(growth, np.abs(piv).max() / np.abs(d).max())
            stab.append({"case": name, "closure": closure, "max_rel_diff_vs_lapack": float(worst),
                         "fraction_diag_dominant": dominant / M, "pivot_growth": float(growth)})
            print("thomas", stab[-1], flush=True)
    out["thomas"] = stab
    return out


# ---------------------------------------------------------------------------------------
def exp_nonuniform():
    """Constant density, solution with a layer of thickness delta at the bottom: uniform vs stretched grids."""
    delta, Nh = 0.25, 33
    out = {"delta": delta, "Nh": Nh, "Nz": [17, 33, 65, 129, 257, 513], "k": [0.0, 2.0, 4.0, 6.0], "linf": {}, "l2": {},
           "dz_min": {}, "dz_max": {}}
    H = mf.VonMises(L, L, a=2.0)
    Zp = lambda Z: mf.z_layer(Z, L, delta)
    for k in out["k"]:
        e2, ei, dmin, dmax = [], [], [], []
        for Nz in out["Nz"]:
            x, y, z = grid(Nh, k, Nz)
            P, _, F = mf.fields(x, y, z, Zp, H)
            p = FFTFDSolver(x, y, z).solve(F)
            e2.append(rel2(p, P))
            ei.append(relinf(p, P))
            dmin.append(float(np.diff(z).min()))
            dmax.append(float(np.diff(z).max()))
        out["l2"][str(k)], out["linf"][str(k)] = e2, ei
        out["dz_min"][str(k)], out["dz_max"][str(k)] = dmin, dmax
        print("nonuniform k=%g linf:" % k, " ".join("%.2e" % e for e in ei), flush=True)
    # The thesis problem (feature in the middle of the domain) on the same grids, as a control
    Hg = mf.Gaussian2D(np.pi, np.pi, 0.5, 0.5)
    out["control"] = {"N": [32, 64, 128, 256], "l2": {}}
    for k in [0.0, 1.0, 2.0]:
        out["control"]["l2"][str(k)] = []
        for N in out["control"]["N"]:
            x, y, z = grid(N, k)
            P, _, F = mf.fields(x, y, z, mf.z_thesis, Hg)
            out["control"]["l2"][str(k)].append(rel2(FFTFDSolver(x, y, z).solve(F), P))
        print("control k=%g l2:" % k, " ".join("%.2e" % e for e in out["control"]["l2"][str(k)]), flush=True)
    return out


# ---------------------------------------------------------------------------------------
def vd_problem(N, theta, strat=0.0, delta=0.75, a=2.0, k=0.0):
    """Hot layer of thickness delta near the bottom (density ratio 1 + theta) on a stratified background."""
    x, y, z = grid(N, k)
    Hp = mf.VonMises(L, L, a=2.0)
    Hr = mf.VonMises(L, L, a=a)
    P, RHO, F = mf.fields(x, y, z, mf.z_thesis, Hp, theta=theta, Zr=lambda Z: mf.z_layer(Z, L, delta), Hr=Hr, strat=strat)
    return x, y, z, P, RHO, F


def run_methods(x, y, z, F, RHO, P, tol=1e-10, max_iter=300):
    res = {}
    beta = mean_stratification(RHO, x, y, z)
    for label, solver in [("poisson", FFTFDSolver(x, y, z)), ("stratified", FFTFDSolver(x, y, z, beta=beta))]:
        for method in ["picard", "gmres"]:
            t0 = time.perf_counter()
            p, info = solver.solve_variable_density(F, RHO, method=method, tol=tol, max_iter=max_iter)
            res[label + "/" + method] = {
                "iterations": int(info["iterations"]), "converged": bool(info["converged"]),
                "error_l2": rel2(p, P) if info["converged"] else None, "time": time.perf_counter() - t0,
                "rate": rate(info["pr_norm"]), "history": [float(v) for v in info["pr_norm"]],
            }
    return res


def exp_vd():
    out = {}
    # (a) spatial convergence of the converged solution and mesh independence of the iteration
    conv = {"N": [16, 32, 64, 128, 256], "theta": [1.0, 4.0, 9.0], "error_l2": {}, "error_linf": {}, "picard_it": {}, "gmres_it": {}}
    for theta in conv["theta"]:
        e2, ei, itp, itg = [], [], [], []
        for N in conv["N"]:
            x, y, z, P, RHO, F = vd_problem(N, theta)
            s = FFTFDSolver(x, y, z)
            p, ip = s.solve_variable_density(F, RHO, method="picard", tol=1e-10)
            _, ig = s.solve_variable_density(F, RHO, method="gmres", tol=1e-10)
            e2.append(rel2(p, P)); ei.append(relinf(p, P)); itp.append(int(ip["iterations"])); itg.append(int(ig["iterations"]))
        conv["error_l2"][str(theta)], conv["error_linf"][str(theta)] = e2, ei
        conv["picard_it"][str(theta)], conv["gmres_it"][str(theta)] = itp, itg
        print("vd conv theta=%g l2:" % theta, " ".join("%.2e" % e for e in e2), "picard it", itp, "gmres it", itg, flush=True)
    out["convergence"] = conv

    # (b) residual histories and rates against the density ratio (N = 128)
    hist = {"N": 128, "theta": [1.0, 4.0, 9.0, 30.0, 100.0], "runs": {}}
    for theta in hist["theta"]:
        x, y, z, P, RHO, F = vd_problem(hist["N"], theta)
        hist["runs"][str(theta)] = run_methods(x, y, z, F, RHO, P)
        bx, by, bz = log_density_gradient(RHO, x, y, z)
        # gamma of the sufficient condition: (2 Lz / pi) max |grad log rho|
        hist.setdefault("gamma", {})[str(theta)] = float(2 * L / np.pi * np.sqrt(bx**2 + by**2 + bz**2).max())
        r = hist["runs"][str(theta)]
        print("vd hist theta=%g:" % theta, {k: (v["iterations"], v["converged"], round(v["rate"], 3)) for k, v in r.items()}, flush=True)
    out["histories"] = hist

    # (c) background stratification Lz / Hs combined with the hot layer (N = 64)
    tab = {"N": 64, "theta": [0.0, 4.0, 30.0], "LzHs": [0.0, 2.0, 5.0, 10.0, 15.0, 20.0], "runs": {}}
    for theta in tab["theta"]:
        for s in tab["LzHs"]:
            if theta == 0.0 and s == 0.0:
                continue
            x, y, z, P, RHO, F = vd_problem(tab["N"], theta, strat=s / L)
            r = run_methods(x, y, z, F, RHO, P, max_iter=400)
            for v in r.values():
                v.pop("history")
            tab["runs"]["%g/%g" % (theta, s)] = r
            print("vd strat theta=%g Lz/Hs=%g:" % (theta, s), {k: (v["iterations"], v["converged"], round(v["rate"], 3)) for k, v in r.items()}, flush=True)
    out["stratification"] = tab

    # (d) sharper hot spot (thin layer, narrow in x-y), with and without vertical stretching
    sharp = {"N": 128, "delta": 0.3, "a": 8.0, "theta": [4.0, 30.0], "runs": {}}
    for theta in sharp["theta"]:
        for k in [0.0, 2.0]:
            x, y, z, P, RHO, F = vd_problem(sharp["N"], theta, delta=sharp["delta"], a=sharp["a"], k=k)
            r = run_methods(x, y, z, F, RHO, P)
            for v in r.values():
                v.pop("history")
            sharp["runs"]["%g/%g" % (theta, k)] = r
            print("vd sharp theta=%g k=%g:" % (theta, k), {kk: (v["iterations"], v["converged"], v["error_l2"]) for kk, v in r.items()}, flush=True)
    out["sharp"] = sharp
    return out


# ---------------------------------------------------------------------------------------
def exp_divform():
    """The paper's form (grad log rho) against the divergence-form splitting of Dong and Shen, both iterated."""
    out = {"N": 64, "cases": [], "max_iter": 2000}
    cases = [(t, 0.0) for t in [1.0, 4.0, 9.0, 30.0, 100.0]] + [(4.0, s) for s in [2.0, 5.0, 10.0, 15.0]]
    for theta, s in cases:
        x, y, z, P, RHO, F = vd_problem(out["N"] + 1, theta, strat=s / L)
        solver = FFTFDSolver(x, y, z)
        row = {"theta": theta, "LzHs": s, "ratio": float(RHO.max() / RHO.min())}
        for form in ["log", "divergence"]:
            for method, kw in [("picard", dict(max_iter=out["max_iter"])), ("gmres", dict(max_iter=out["max_iter"], restart=50))]:
                t0 = time.perf_counter()
                p, info = solver.solve_variable_density(F, RHO, method=method, form=form, tol=1e-10, **kw)
                row["%s/%s" % (form, method)] = {"iterations": int(info["iterations"]), "converged": bool(info["converged"]),
                                                 "error_l2": rel2(p, P), "time": time.perf_counter() - t0}
        out["cases"].append(row)
        print("divform theta=%g Lz/Hs=%g ratio=%.3g:" % (theta, s, row["ratio"]),
              {k: (v["iterations"], v["converged"]) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    return out


# ---------------------------------------------------------------------------------------
def best_time(fun, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fun()
        best = min(best, time.perf_counter() - t0)
    return best


def timed_solve(x, y, z, F, threads, repeats=3):
    set_threads(threads)
    s = FFTFDSolver(x, y, z, workers=threads)
    s.solve(F)  # warm-up (JIT compilation, FFT plans)
    best = None
    for _ in range(repeats):
        s.timings = {"fft": 0.0, "tridiagonal": 0.0, "rhs": 0.0, "solves": 0}
        t0 = time.perf_counter()
        s.solve(F)
        total = time.perf_counter() - t0
        if best is None or total < best["total"]:
            best = {"total": total, "fft": s.timings["fft"], "tridiagonal": s.timings["tridiagonal"]}
    return best


def exp_perf():
    from fftfd_original import fftfd as fftfd_loop  # original implementation: Python loop over the modes

    out = {"machine": platform.processor() + " " + platform.machine(), "max_threads": MAX_THREADS,
           "N": [33, 65, 129, 257, 513], "loop": [], "vector_1": [], "vector_all": []}
    H = mf.Gaussian2D(np.pi, np.pi, 0.5, 0.5)
    for N in out["N"]:
        x, y, z = grid(N)
        _, _, F = mf.fields(x, y, z, mf.z_thesis, H)
        if N <= 257:
            fftfd_loop(F.copy(), x, y, z)
            out["loop"].append(best_time(lambda: fftfd_loop(F.copy(), x, y, z), 3 if N < 257 else 2))
        else:
            out["loop"].append(None)
        out["vector_1"].append(timed_solve(x, y, z, F, 1))
        out["vector_all"].append(timed_solve(x, y, z, F, MAX_THREADS))
        print("perf N=%d loop=%s  1 thread=%.4f  %d threads=%.4f" % (N, out["loop"][-1], out["vector_1"][-1]["total"], MAX_THREADS, out["vector_all"][-1]["total"]), flush=True)

    # FFT length N - 1: 2^n - 1 (prime or with large prime factors) against 2^n
    out["fft_length"] = {"N": [128, 129, 256, 257, 512, 513], "runs": []}
    for N in out["fft_length"]["N"]:
        x, y, z = grid(N)
        _, _, F = mf.fields(x, y, z, mf.z_thesis, H)
        out["fft_length"]["runs"].append(timed_solve(x, y, z, F, 1))
        print("fft length N=%d" % N, out["fft_length"]["runs"][-1], flush=True)

    out["scaling"] = {}
    for N in [257, 513]:
        x, y, z = grid(N)
        _, _, F = mf.fields(x, y, z, mf.z_thesis, H)
        threads = [t for t in [1, 2, 4, 6, 8, 10] if t <= MAX_THREADS]
        runs = [timed_solve(x, y, z, F, t, repeats=5) for t in threads]
        out["scaling"][str(N)] = {"threads": threads, "runs": runs}
        print("scaling N=%d" % N, [(t, round(r["total"], 4)) for t, r in zip(threads, runs)], flush=True)

    # cost of a variable-density solve relative to one Poisson solve
    set_threads(MAX_THREADS)
    N = 257
    x, y, z, P, RHO, F = vd_problem(N, 4.0)
    one = timed_solve(x, y, z, F, MAX_THREADS)["total"]
    vd = {"N": N, "theta": 4.0, "poisson_solve": one}
    for method in ["picard", "gmres"]:
        s = FFTFDSolver(x, y, z, workers=MAX_THREADS)
        s.solve_variable_density(F, RHO, method=method, tol=1e-10, max_iter=2)  # warm-up
        s.timings = {"fft": 0.0, "tridiagonal": 0.0, "rhs": 0.0, "solves": 0}
        t0 = time.perf_counter()
        _, info = s.solve_variable_density(F, RHO, method=method, tol=1e-10)
        vd[method] = {"time": time.perf_counter() - t0, "iterations": int(info["iterations"]), "solves": s.timings["solves"],
                      "fft": s.timings["fft"], "tridiagonal": s.timings["tridiagonal"], "rhs": s.timings["rhs"]}
    out["variable_density"] = vd
    print("perf vd", vd, flush=True)
    return out


if __name__ == "__main__":
    groups = {"const": exp_const, "nonuniform": exp_nonuniform, "vd": exp_vd, "divform": exp_divform, "perf": exp_perf}
    selected = sys.argv[1:] or list(groups)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in selected:
        t0 = time.perf_counter()
        results[name] = groups[name]()
        print("== %s done in %.1f s" % (name, time.perf_counter() - t0), flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(results, open(OUT, "w"), indent=1)
