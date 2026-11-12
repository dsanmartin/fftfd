"""LaTeX tables for the paper, from paper/data/results.json (run experiments.py first)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "paper", "data", "results.json")))
TAB = os.path.join(HERE, "paper", "tables")
os.makedirs(TAB, exist_ok=True)


def sci(v, digits=2):
    """1.23e-04 -> $1.23\\times10^{-4}$"""
    if v is None or not np.isfinite(v):
        return "--"
    m, e = ("%.*e" % (digits, v)).split("e")
    return r"$%s\times10^{%d}$" % (m, int(e))


def orders(err):
    return ["--"] + ["%.2f" % np.log2(err[i - 1] / err[i]) for i in range(1, len(err))]


def write(name, lines):
    with open(os.path.join(TAB, name + ".tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def tab_const():
    c = R["const"]
    e2, ei = c["one-sided"]["l2"], c["one-sided"]["linf"]
    o2, oi = orders(e2), orders(ei)
    rows = [r"\begin{tabular}{rcccc}", r"\toprule",
            r"$N$ & $\ell_2$ error & order & $\ell_\infty$ error & order \\", r"\midrule"]
    for i, N in enumerate(c["N"]):
        rows.append("%d & %s & %s & %s & %s \\\\" % (N - 1, sci(e2[i]), o2[i], sci(ei[i]), oi[i]))
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("const", rows)


def tab_thomas():
    names = {"dz = dx": r"$\Delta z=\Delta x$", "dz = dx / 16": r"$\Delta z=\Delta x/16$", "dz = 16 dx": r"$\Delta z=16\,\Delta x$",
             "stretched, k = 4": r"stretched, $k_s=4$"}
    rows = [r"\begin{tabular}{llccc}", r"\toprule",
            r"Grid & Bottom closure & Diag.\ dominant modes & Pivot growth & Difference vs.\ pivoted LU \\", r"\midrule"]
    for t in R["const"]["thomas"]:
        rows.append("%s & %s & %.1f\\,\\%% & %.2f & %s \\\\" % (names[t["case"]], t["closure"], 100 * t["fraction_diag_dominant"],
                                                              t["pivot_growth"], sci(t["max_rel_diff_vs_lapack"], 1)))
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("thomas", rows)


def tab_nonuniform():
    n = R["nonuniform"]
    rows = [r"\begin{tabular}{r" + "c" * len(n["k"]) + "}", r"\toprule",
            r"$N_z$ & " + " & ".join("uniform" if k == 0 else "$k_s=%g$" % k for k in n["k"]) + r" \\", r"\midrule"]
    for i, Nz in enumerate(n["Nz"]):
        rows.append("%d & " % (Nz - 1) + " & ".join(sci(n["linf"][str(k)][i], 1) for k in n["k"]) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("nonuniform", rows)


def tab_vd():
    v = R["vd"]["convergence"]
    th = v["theta"]
    head = " & ".join(r"\multicolumn{3}{c}{$\rho_{\max}/\rho_{\min}=%d$}" % round(1 + t) for t in th)
    rows = [r"\begin{tabular}{r" + "ccc" * len(th) + "}", r"\toprule", r" & " + head + r" \\",
            " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(th))),
            r"$N$ & " + " & ".join([r"$\ell_2$ error & FP & GMRES"] * len(th)) + r" \\", r"\midrule"]
    for i, N in enumerate(v["N"]):
        cells = []
        for t in th:
            cells += [sci(v["error_l2"][str(t)][i]), "%d" % v["picard_it"][str(t)][i], "%d" % v["gmres_it"][str(t)][i]]
        rows.append("%d & " % (N - 1) + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("vd_convergence", rows)


def tab_strat():
    t = R["vd"]["stratification"]
    keys = ["poisson/picard", "poisson/gmres", "stratified/picard", "stratified/gmres"]
    rows = [r"\begin{tabular}{ccrrrr}", r"\toprule",
            r" & & \multicolumn{2}{c}{Poisson operator $\mathcal{L}_0$} & \multicolumn{2}{c}{Stratified operator $\mathcal{L}_\beta$} \\",
            r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
            r"$\rho_{\max}/\rho_{\min}$ (layer) & $L_z/H_\rho$ & FP & GMRES & FP & GMRES \\", r"\midrule"]
    for theta in t["theta"]:
        for s in t["LzHs"]:
            key = "%g/%g" % (theta, s)
            if key not in t["runs"]:
                continue
            cells = []
            for k in keys:
                r_ = t["runs"][key][k]
                cells.append("%d" % r_["iterations"] if r_["converged"] else "div.")
            rows.append("%d & %g & " % (round(1 + theta), s) + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    rows[-1] = r"\bottomrule"
    rows.append(r"\end{tabular}")
    write("stratification", rows)


def tab_perf():
    p = R["perf"]
    rows = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
            r" & & \multicolumn{3}{c}{Batched, 1 thread} & \multicolumn{2}{c}{Batched, %d threads} \\" % p["max_threads"],
            r"\cmidrule(lr){3-5} \cmidrule(lr){6-7}",
            r"$N$ & Loop & Total & FFT & Tridiag. & Total & Speed-up \\", r"\midrule"]
    for i, N in enumerate(p["N"]):
        v1, va, lp = p["vector_1"][i], p["vector_all"][i], p["loop"][i]
        rows.append("%d & %s & %.4f & %.4f & %.4f & %.4f & %.1f \\\\" % (
            N - 1, "--" if lp is None else "%.4f" % lp, v1["total"], v1["fft"], v1["tridiagonal"], va["total"], v1["total"] / va["total"]))
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("performance", rows)


def tab_fds():
    path = os.path.join(HERE, "paper", "data", "fds_replay.json")
    if not os.path.exists(path):
        return
    r = json.load(open(path))["runs"]
    rows = [r"\begin{tabular}{llcccc}", r"\toprule",
            r" & & \multicolumn{2}{c}{$\varepsilon=10^{-6}$} & \multicolumn{2}{c}{$\varepsilon=10^{-10}$} \\",
            r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
            r"Method & Initial guess & mean & range & mean & range \\", r"\midrule"]
    for method, name in [("picard", "Fixed point"), ("gmres", "GMRES")]:
        for start, label in [("cold", "zero"), ("warm", "previous step")]:
            cells = []
            for tol in ["1e-06", "1e-10"]:
                it = np.array(r["poisson/%s/%s/%s" % (method, start, tol)]["iterations"])
                it = it[1:] if start == "warm" else it
                cells += ["%.1f" % it.mean(), "%d--%d" % (it.min(), it.max())]
            rows.append("%s & %s & " % (name, label) + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("fds", rows)


def tab_divform():
    d = R.get("divform")
    if d is None:
        return
    rows = [r"\begin{tabular}{rrcccccc}", r"\toprule",
            r" & & \multicolumn{3}{c}{Form $\nabla\ln\rho\cdot\nabla p$} & \multicolumn{3}{c}{Divergence form \eqref{eq:divergence-form}} \\",
            r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
            r"$\rho_{\max}/\rho_{\min}$ & $L_z/H_\rho$ & FP & GMRES & $\ell_2$ error & FP & GMRES & $\ell_2$ error \\", r"\midrule"]
    cap = d["max_iter"]
    it = lambda v: "%d" % v["iterations"] if v["converged"] else "n.c."
    for c in d["cases"]:
        err = lambda f: next((sci(c[f + "/" + m]["error_l2"], 1) for m in ["gmres", "picard"] if c[f + "/" + m]["converged"]), "--")
        ratio = "%d" % round(c["ratio"]) if c["ratio"] < 1e3 else sci(c["ratio"], 1)
        rows.append("%s & %g & %s & %s & %s & %s & %s & %s \\\\" % (ratio, c["LzHs"], it(c["log/picard"]), it(c["log/gmres"]), err("log"),
                                                                 it(c["divergence/picard"]), it(c["divergence/gmres"]), err("divergence")))
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("divform", rows)


def numbers():
    """Timing numbers quoted in the text, as LaTeX macros (timings change slightly from run to run)."""
    p = R["perf"]
    ratios = [l / v["total"] for l, v in zip(p["loop"], p["vector_1"]) if l is not None]
    sc = {k: p["scaling"][k]["runs"] for k in sorted(p["scaling"], key=int)}
    small, large = sc[min(sc, key=int)], sc[max(sc, key=int)]
    fl = dict(zip(p["fft_length"]["N"], p["fft_length"]["runs"]))
    vd = p["variable_density"]
    last = p["vector_1"][-1]
    macros = {
        "perfLoopMin": "%.0f" % min(ratios), "perfLoopMax": "%.0f" % max(ratios),
        "perfSpeedSmall": "%.1f" % (small[0]["total"] / small[-1]["total"]),
        "perfSpeedLarge": "%.1f" % (large[0]["total"] / large[-1]["total"]),
        "perfSpeedMax": "%.1f" % max(small[0]["total"] / small[-1]["total"], large[0]["total"] / large[-1]["total"]),
        "perfSpeedTri": "%.1f" % (large[0]["tridiagonal"] / large[-1]["tridiagonal"]),
        "perfSpeedFft": "%.1f" % (large[0]["fft"] / large[-1]["fft"]),
        "perfFftShare": "%.0f" % (100 * last["fft"] / last["total"]),
        "perfTriShare": "%.0f" % (100 * last["tridiagonal"] / last["total"]),
        "fftPrimeSmall": "%.3f" % fl[128]["total"], "fftPowerSmall": "%.3f" % fl[129]["total"],
        "fftPrimeLarge": "%.1f" % fl[512]["total"], "fftPowerLarge": "%.1f" % fl[513]["total"],
        "vdPicardTime": "%.1f" % vd["picard"]["time"], "vdGmresTime": "%.1f" % vd["gmres"]["time"],
        "vdPicardIts": "%d" % vd["picard"]["iterations"], "vdGmresIts": "%d" % vd["gmres"]["iterations"],
        "vdGmresSolves": "%d" % vd["gmres"]["solves"], "vdSolveTime": "%.3f" % vd["poisson_solve"],
        "vdCostRatio": "%.1f" % (vd["picard"]["time"] / vd["picard"]["iterations"] / vd["poisson_solve"]),
    }
    write("numbers", ["% Generated by tables.py from paper/data/results.json"] +
          ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in macros.items()])
    return macros


if __name__ == "__main__":
    tab_const(); tab_thomas(); tab_nonuniform(); tab_vd(); tab_strat(); tab_perf(); tab_fds(); tab_divform()
    for k, v in numbers().items():
        print("  %-16s %s" % (k, v))
    print("tables written to", TAB)
