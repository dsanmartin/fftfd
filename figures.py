"""Figures for the paper, from paper/data/results.json (run experiments.py first)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solver import stretched_grid

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "paper", "data", "results.json")))
FIG = os.path.join(HERE, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

# Categorical slots (fixed order) and a one-hue ramp for ordered parameters
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
MARK = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    # Text is typeset by LaTeX in Computer Modern, the default font of the paper
    "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def reference(ax, xs, ys, order, label, anchor=0):
    """Dashed power-law guide through (xs[anchor], ys[anchor])."""
    xs = np.asarray(xs, dtype=float)
    ax.plot(xs, ys[anchor] * (xs / xs[anchor]) ** order, "--", color=MUTED, lw=0.9, label=label, zorder=1)


# ----------------------------------------------------------------------------------------
def fig_convergence():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    c = R["const"]
    dz = np.array(c["dz"])
    ax = axs[0]
    ax.loglog(dz, c["one-sided"]["l2"], "-", marker=MARK[0], color=C[0], label=r"relative $\ell_2$ error")
    ax.loglog(dz, c["one-sided"]["linf"], "-", marker=MARK[1], color=C[1], label=r"relative $\ell_\infty$ error")
    reference(ax, dz, 1.6 * np.array(c["one-sided"]["linf"]), 2, r"$\mathcal{O}(\Delta z^2)$")
    ax.set_xlabel(r"$\Delta z$")
    ax.set_ylabel("error")
    ax.set_title("(a)", loc="left")
    ax.legend(loc="lower right")

    v = R["vd"]["convergence"]
    dzv = 2 * np.pi / (np.array(v["N"]) - 1)
    ax = axs[1]
    for i, th in enumerate(v["theta"]):
        ax.loglog(dzv, v["error_l2"][str(th)], "-", marker=MARK[i], color=C[i],
                  label=r"$\rho_{\max}/\rho_{\min}=%d$" % round(1 + th))
    reference(ax, dzv, 1.8 * np.array(v["error_l2"][str(v["theta"][-1])]), 2, r"$\mathcal{O}(\Delta z^2)$")
    ax.set_xlabel(r"$\Delta z$")
    ax.set_title("(b)", loc="left")
    ax.legend(loc="lower right")
    for ax in axs:
        ax.set_xscale("log", base=2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "convergence")


def fig_nonuniform():
    n = R["nonuniform"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1, 1.55]})
    ax = axs[0]
    for i, k in enumerate(n["k"]):
        z = stretched_grid(0.0, 2 * np.pi, 33, k)
        ax.plot(np.full_like(z, i), z, linestyle="none", marker="_", markersize=14, markeredgewidth=0.8, color=RAMP[i + 1])
    ax.axhspan(0, n["delta"], color=GRID, zorder=0)
    ax.text(len(n["k"]) - 0.45, n["delta"] / 2, r"$\delta$", va="center", ha="left", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(n["k"])))
    ax.set_xticklabels(["%g" % k for k in n["k"]])
    ax.set_xlim(-0.6, len(n["k"]) - 0.2)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_xlabel(r"stretching parameter $k_s$")
    ax.set_ylabel(r"$z$")
    ax.set_title("(a)", loc="left")
    ax.grid(False)

    ax = axs[1]
    Nz = np.array(n["Nz"])
    for i, k in enumerate(n["k"]):
        ax.loglog(Nz - 1, n["linf"][str(k)], "-", marker=MARK[i], color=RAMP[i + 1],
                  label=("uniform" if k == 0 else r"$k_s=%g$" % k))
    reference(ax, Nz - 1, 0.45 * np.array(n["linf"]["6.0"]), -2, r"$\mathcal{O}(N_z^{-2})$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"vertical intervals $N_z$")
    ax.set_ylabel(r"relative $\ell_\infty$ error")
    ax.set_title("(b)", loc="left")
    ax.set_ylim(5e-6, 2e3)
    ax.legend(loc="upper right", ncol=3, columnspacing=1.0, handlelength=1.6, fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "nonuniform")


def fig_iterations():
    h = R["vd"]["histories"]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, method, title in [(axs[0], "poisson/picard", "(a)"), (axs[1], "poisson/gmres", "(b)")]:
        for i, th in enumerate(h["theta"]):
            r = np.array(h["runs"][str(th)][method]["history"])
            ax.semilogy(np.arange(1, len(r) + 1), r, "-", marker=MARK[i], markersize=3, color=RAMP[i],
                        label="%d" % round(1 + th))
        ax.set_xlabel("iteration")
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 38)
        ax.set_ylim(1e-11, 3)
    axs[0].set_ylabel("relative residual")
    axs[1].set_yticklabels([])
    axs[1].legend(title=r"$\rho_{\max}/\rho_{\min}$", loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.4)

    ax = axs[2]
    t = R["vd"]["stratification"]
    s = np.array(t["LzHs"])
    series = [("poisson/picard", "fixed point", C[0], MARK[0]), ("poisson/gmres", "GMRES", C[1], MARK[1]),
              ("stratified/picard", "fixed point, stratified", C[2], MARK[2]), ("stratified/gmres", "GMRES, stratified", C[3], MARK[3])]
    for key, label, col, mk in series:
        its = [t["runs"]["4/%g" % v][key]["iterations"] if t["runs"]["4/%g" % v][key]["converged"] else np.nan for v in s]
        ax.plot(s, its, "-", marker=mk, color=col, label=label)
    ax.annotate("diverges", xy=(11.3, 78), xytext=(12.6, 58), color=MUTED, fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.set_ylim(0, 80)
    ax.set_xlabel(r"stratification $L_z/H_\rho$")
    ax.set_ylabel("iterations")
    ax.set_title("(c)", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0.09, 1, 1))
    fig.legend(handles, labels, title="panel (c):", loc="lower right", ncol=4, fontsize=7, title_fontsize=7,
               handlelength=1.4, columnspacing=1.0, bbox_to_anchor=(0.995, 0.0), alignment="left")
    save(fig, "iterations")


def fig_performance():
    p = R["perf"]
    N = np.array(p["N"])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axs[0]
    loop = np.array([np.nan if v is None else v for v in p["loop"]], dtype=float)
    t1 = np.array([v["total"] for v in p["vector_1"]])
    ta = np.array([v["total"] for v in p["vector_all"]])
    ax.loglog(N - 1, loop, "-", marker=MARK[0], color=C[0], label="loop over modes (original)")
    ax.loglog(N - 1, t1, "-", marker=MARK[1], color=C[1], label="batched, 1 thread")
    ax.loglog(N - 1, ta, "-", marker=MARK[2], color=C[2], label="batched, %d threads" % p["max_threads"])
    n = (N - 1).astype(float)
    ref = n**3 * np.log2(n**2)
    ax.loglog(n, ref * t1[-1] / ref[-1] * 2.5, "--", color=MUTED, lw=0.9, label=r"$\mathcal{O}(N^3\log N^2)$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$N$ (intervals per direction)")
    ax.set_ylabel("time per solve (s)")
    ax.set_title("(a)", loc="left")
    ax.legend(loc="upper left", fontsize=7)

    ax = axs[1]
    for i, key in enumerate(sorted(p["scaling"], key=int)):
        sc = p["scaling"][key]
        th = np.array(sc["threads"])
        tot = np.array([r["total"] for r in sc["runs"]])
        ax.plot(th, tot[0] / tot, "-", marker=MARK[i], color=C[i], label=r"$N=%d$" % (int(key) - 1))
    sc = p["scaling"][max(p["scaling"], key=int)]
    th = np.array(sc["threads"])
    tri = np.array([r["tridiagonal"] for r in sc["runs"]])
    ax.plot(th, tri[0] / tri, ":", marker=MARK[2], color=C[2], label=r"tridiagonal stage only, $N=%d$" % (int(max(p["scaling"], key=int)) - 1))
    ax.plot(th, th, "--", color=MUTED, lw=0.9, label="ideal")
    ax.set_xticks(th)
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("threads")
    ax.set_ylabel("speed-up")
    ax.set_title("(b)", loc="left")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "performance")


# ----------------------------------------------------------------------------------------
# Snapshots of the fields
# ----------------------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap

SEQ = LinearSegmentedColormap.from_list("seq", ["#f7fafd", "#b7d3f6", "#5598e7", "#256abf", "#0d366b"])
HEAT = LinearSegmentedColormap.from_list("heat", ["#fff6ec", "#f8c48f", "#eb6834", "#a8320f", "#4d1300"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3b08f", "#eb6834", "#8a2d0b"])


def panel(fig, ax, X, Z, F, cmap, label, symmetric=False, **kw):
    if symmetric:
        m = np.abs(F).max()
        kw.update(vmin=-m, vmax=m)
    im = ax.pcolormesh(X, Z, F, cmap=cmap, shading="gouraud", rasterized=True, **kw)
    cb = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.05)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.4)
    cb.formatter.set_powerlimits((-2, 3))
    cb.ax.yaxis.get_offset_text().set_fontsize(7)
    cb.update_ticks()
    ax.set_title(label, loc="left")
    ax.grid(False)
    return im


def fig_fields_manufactured():
    """Exact solution, numerical solution and error on the plane y = pi for the three test problems."""
    import manufactured as mf
    from solver import FFTFDSolver

    L = 2 * np.pi
    cases = []
    # (i) constant density, uniform grid
    x = np.linspace(0, L, 65)
    z = np.linspace(0, L, 65)
    P, _, F = mf.fields(x, x, z, mf.z_thesis, mf.Gaussian2D(np.pi, np.pi, 0.5, 0.5))
    cases.append((x, z, P, FFTFDSolver(x, x, z).solve(F), None, None))
    # (ii) bottom layer, stretched grid
    xh = np.linspace(0, L, 33)
    zs = stretched_grid(0.0, L, 65, 4.0)
    H = mf.VonMises(L, L, a=2.0)
    P, _, F = mf.fields(xh, xh, zs, lambda Z: mf.z_layer(Z, L, 0.25), H)
    cases.append((xh, zs, P, FFTFDSolver(xh, xh, zs).solve(F), None, 1.6))
    # (iii) variable density, density ratio 5
    P, RHO, F = mf.fields(x, x, z, mf.z_thesis, H, theta=4.0, Zr=lambda Z: mf.z_layer(Z, L, 0.75), Hr=H)
    ph, _ = FFTFDSolver(x, x, z).solve_variable_density(F, RHO, method="gmres", tol=1e-10)
    cases.append((x, z, P, ph, RHO, None))

    fig, axs = plt.subplots(3, 3, figsize=(6.6, 5.9), sharex="col")
    rows = ["(a)", "(b)", "(c)"]
    for i, (xc, zc, P, Ph, RHO, zmax) in enumerate(cases):
        j = xc.shape[0] // 2  # plane y = pi
        X, Z = np.meshgrid(xc, zc, indexing="ij")
        vmin, vmax = min(P[:, j].min(), 0.0), P[:, j].max()
        panel(fig, axs[i, 0], X, Z, P[:, j], SEQ, rows[i], vmin=vmin, vmax=vmax)
        panel(fig, axs[i, 1], X, Z, Ph[:, j], SEQ, "", vmin=vmin, vmax=vmax)
        panel(fig, axs[i, 2], X, Z, Ph[:, j] - P[:, j], DIV, "", symmetric=True)
        if RHO is not None:
            cs = axs[i, 0].contour(X, Z, RHO[:, j], levels=[0.3, 0.5, 0.7, 0.9], colors=C[1], linewidths=0.7)
            axs[i, 0].clabel(cs, fmt="%.1f", fontsize=6)
        for ax in axs[i]:
            ax.set_aspect("equal" if zmax is None else "auto")
            if zmax is not None:
                ax.set_ylim(0, zmax)
        axs[i, 0].set_ylabel(r"$z$")
    for ax in axs[-1]:
        ax.set_xlabel(r"$x$")
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    save(fig, "fields_manufactured")


def fds_snapshot(t_target=6.2):
    """x-z slice through the burner of the FDS fields at one time (cached in paper/data)."""
    cache = os.path.join(HERE, "paper", "data", "fds_snapshot.npz")
    if not os.path.exists(cache):
        from fds_replay import load_fields
        fields, times = load_fields(os.path.join(HERE, "fds"), "flame_periodic_mpi", required=("DENSITY", "PRESSURE", "TEMPERATURE"))
        n = int(np.argmin(np.abs(times - t_target)))
        j = fields["DENSITY"].shape[2] // 2
        np.savez_compressed(cache, time=times[n], y_index=j, tmax_all=fields["TEMPERATURE"].max(),
                            **{k.lower(): v[n, :, j, :] for k, v in fields.items()})
    return np.load(cache)


def fig_fds_fields():
    d = fds_snapshot()
    n = d["density"].shape[0]
    c = (np.arange(n) + 0.5) * 1.92 / n  # cell centres
    X, Z = np.meshgrid(c, c, indexing="ij")
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=True)
    panel(fig, axs[0], X, Z, d["temperature"], HEAT, "(a)")
    panel(fig, axs[1], X, Z, d["density"], SEQ, "(b)")
    panel(fig, axs[2], X, Z, d["pressure"], DIV, "(c)", symmetric=True)
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ (m)")
    axs[0].set_ylabel(r"$z$ (m)")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fds_fields")


def fig_fds_iterations():
    r = json.load(open(os.path.join(HERE, "paper", "data", "fds_replay.json")))
    t = np.array(r["times"])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axs[0]
    series = [("poisson/picard/cold/1e-06", "FP, cold", C[0], "-"), ("poisson/picard/warm/1e-06", "FP, warm", C[0], ":"),
              ("poisson/gmres/cold/1e-06", "GMRES, cold", C[1], "-"), ("poisson/gmres/warm/1e-06", "GMRES, warm", C[1], ":")]
    for key, label, col, ls in series:
        it = np.array(r["runs"][key]["iterations"], dtype=float)
        if "/warm/" in key:
            it[0] = np.nan  # no previous step
        ax.plot(t, it, ls, color=col, lw=1.1, drawstyle="steps-mid", label=label)
    ax.set_ylim(0, 24)
    ax.set_xlabel(r"time (s)")
    ax.set_ylabel("iterations per time step")
    ax.set_title("(a)", loc="left")
    ax.legend(loc="upper center", ncol=2, fontsize=7, columnspacing=1.2, handlelength=1.8)
    ax = axs[1]
    ax.semilogy(t[1:], r["lagged_error"], "-", color=C[1], lw=1.1, label="error after one lagged sweep")
    ax.semilogy(t[1:], r["step_change"], "-", color=C[0], lw=1.1, label="pressure change per time step")
    ax.set_ylim(1e-2, 30)
    ax.set_xlabel(r"time (s)")
    ax.set_ylabel("relative difference")
    ax.set_title("(b)", loc="left")
    ax.legend(loc="upper center", fontsize=7, handlelength=1.8)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fds_iterations")


if __name__ == "__main__":
    import sys
    if "fds" in sys.argv[1:]:
        fig_fds_iterations()
        sys.exit()
    fig_convergence()
    fig_nonuniform()
    fig_iterations()
    fig_performance()
    fig_fields_manufactured()
    fig_fds_fields()
    fig_fds_iterations()
    print("figures written to", FIG)
