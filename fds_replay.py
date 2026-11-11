"""
Replay of FDS fields through the variable-density pressure solver.

    python fds_replay.py <folder with the FDS output> [CHID] [Lx Ly Lz]

For every saved time step the FDS density rho and perturbation pressure p are read (cell-centred
3D slices), a right-hand side is built by applying the discrete operator to the FDS pressure,

    f = lap(p_FDS) - grad(rho)/rho . grad(p_FDS),   p_top = p_FDS at the top cells,
    dp/dz at the bottom = one-sided difference of p_FDS,

so that the FDS pressure is the exact solution of the discrete problem,

and the problem is solved with each variant (fixed point / GMRES, Poisson / stratified operator),
from a cold start (p0 = 0) and from a warm start (solution of the previous time step).
The cell centres are used as the nodes of the solver grid. Writes paper/data/fds_replay.json.
"""
import glob
import json
import os
import re
import struct
import sys

import numpy as np
import scipy.fft as sfft

from solver import FFTFDSolver, _add_density_term, log_density_gradient, mean_stratification


def read_sf(path):
    """FDS slice file -> (quantity, index bounds, times, frames[nt, ni, nj, nk])."""
    with open(path, "rb") as fh:
        def rec():
            n = struct.unpack("i", fh.read(4))[0]
            d = fh.read(n)
            fh.read(4)
            return d
        names = [rec().decode().strip() for _ in range(3)]
        b = struct.unpack("6i", rec())
        shape = (b[1] - b[0] + 1, b[3] - b[2] + 1, b[5] - b[4] + 1)
        times, frames = [], []
        while True:
            if len(fh.read(4)) < 4:
                break
            t = struct.unpack("f", fh.read(4))[0]
            fh.read(4)
            d = rec()
            if len(d) < 4 * shape[0] * shape[1] * shape[2]:
                break  # incomplete last frame
            times.append(t)
            frames.append(np.frombuffer(d, dtype=np.float32).reshape(shape, order="F"))
    return names[0], b, np.array(times), np.array(frames)


def read_sf_header(path):
    """Read the quantity and index bounds without loading the time frames."""
    with open(path, "rb") as fh:
        def rec():
            n = struct.unpack("i", fh.read(4))[0]
            d = fh.read(n)
            fh.read(4)
            return d
        names = [rec().decode().strip() for _ in range(3)]
        bounds = struct.unpack("6i", rec())
    return names[0], bounds


def read_smv_meshes(folder, chid):
    """Return mesh origins and cell counts from the Smokeview metadata."""
    path = os.path.join(folder, chid + ".smv")
    if not os.path.isfile(path):
        raise ValueError("Cannot stitch MPI slices without the mesh metadata file %s" % path)
    meshes = {}
    with open(path, encoding="latin-1") as fh:
        lines = iter(fh)
        for line in lines:
            match = re.match(r"\s*GRID\s+MESH_(\d+)\s*$", line)
            if not match:
                continue
            mesh_id = int(match.group(1))
            dims = tuple(int(v) for v in next(lines).split()[:3])
            while True:
                line = next(lines)
                if line.strip() == "PDIM":
                    p = [float(v) for v in next(lines).split()[:6]]
                    break
            meshes[mesh_id] = ((p[0], p[2], p[4]), dims,
                               (p[1], p[3], p[5]))
    if not meshes:
        raise ValueError("No mesh geometry found in %s" % path)
    return meshes


def load_fields(folder, chid, required=("DENSITY", "PRESSURE")):
    """Read and stitch cell-centred fields from a single-mesh or MPI FDS run."""
    required = set(required)
    pattern = re.compile(r"%s_\d+_\d+\.sf$" % re.escape(chid))
    candidates = sorted(path for path in glob.glob(os.path.join(folder, chid + "_*.sf"))
                        if pattern.match(os.path.basename(path)))
    if not candidates:
        nested = sorted(glob.glob(os.path.join(folder, "*", chid + "_*.sf")))
        folders = sorted({os.path.dirname(path) for path in nested})
        if len(folders) == 1:
            folder = folders[0]
            candidates = sorted(path for path in glob.glob(os.path.join(folder, chid + "_*.sf"))
                                if pattern.match(os.path.basename(path)))

    # Gather only full 3D volume slices. Each MPI mesh writes its own copy of
    # the requested full-domain slice, with local indices in the SF header.
    volume_files = {}
    for path in candidates:
        match = re.match(r"%s_(\d+)_(\d+)\.sf$" % re.escape(chid), os.path.basename(path))
        if not match:
            continue
        mesh_id = int(match.group(1))
        name, b = read_sf_header(path)
        if all(b[2 * axis] != b[2 * axis + 1] for axis in range(3)):
            volume_files.setdefault(name, {}).setdefault(mesh_id, path)

    if not required.issubset(volume_files):
        return {}, None

    meshes = read_smv_meshes(folder, chid)
    mesh_ids = sorted(set.intersection(*(set(volume_files[name]) for name in required)))
    if not mesh_ids:
        return {}, None

    # Build global cell offsets from each mesh's physical origin. FDS stores
    # shared mesh faces in the SF records; dropping index zero leaves each
    # mesh's 32 owned cell centres, with no duplicated interface cells.
    origins = np.array([meshes[mid][0] for mid in mesh_ids], dtype=float)
    dims_by_mesh = {mid: meshes[mid][1] for mid in mesh_ids}
    spacing = np.array([
        (meshes[mesh_ids[0]][2][axis] - meshes[mesh_ids[0]][0][axis])
        / dims_by_mesh[mesh_ids[0]][axis] for axis in range(3)
    ])
    global_origin = np.min(origins, axis=0)
    offsets = {mid: np.rint((np.array(meshes[mid][0]) - global_origin) / spacing).astype(int)
               for mid in mesh_ids}
    shape = tuple(max(offsets[mid][axis] + dims_by_mesh[mid][axis] for mid in mesh_ids)
                  for axis in range(3))

    out, times = {}, None
    for name in required:
        full = None
        written = np.zeros(shape, dtype=bool)
        for mid in mesh_ids:
            field_name, b, t, fr = read_sf(volume_files[name][mid])
            expected = tuple(dims_by_mesh[mid][axis] + 1 for axis in range(3))
            if fr.ndim != 4 or fr.shape[1:] != expected:
                raise ValueError("Unexpected %s slice shape in %s: %s (expected %s)" %
                                 (name, volume_files[name][mid], fr.shape[1:], expected))
            if times is not None and (len(t) != len(times) or not np.allclose(t, times)):
                raise ValueError("FDS field files have different time records; cannot replay together")
            times = t
            if full is None:
                full = np.empty((len(t),) + shape, dtype=np.float32)
            off = offsets[mid]
            sl = tuple(slice(off[a], off[a] + dims_by_mesh[mid][a]) for a in range(3))
            full[(slice(None),) + sl] = fr[:, 1:, 1:, 1:]
            written[sl] = True
        if not written.all():
            raise ValueError("MPI %s slices do not cover the full mesh domain" % name)
        out[name] = full.astype(float)
    return out, times


def pad(a):
    """(nx, ny, nz) periodic block -> (nx + 1, ny + 1, nz) with the periodic images."""
    out = np.empty((a.shape[0] + 1, a.shape[1] + 1, a.shape[2]))
    out[:-1, :-1] = a
    out[-1, :-1] = a[0]
    out[:, -1] = out[:, 0]
    return out


def apply_operator(s, p, rho):
    """Discrete lap(p) - grad(rho)/rho . grad(p) with the differences of the solver."""
    pb = np.ascontiguousarray(p[:-1, :-1, :])
    f = sfft.irfft2(-s.kappa2d[:, :, None] * sfft.rfft2(pb, axes=(0, 1)), s=(s.nx, s.ny), axes=(0, 1))
    d2 = s.d2
    f[:, :, 1:-1] += d2[0, 1:-1] * pb[:, :, :-2] + d2[1, 1:-1] * pb[:, :, 1:-1] + d2[2, 1:-1] * pb[:, :, 2:]
    bx, by, bz = log_density_gradient(rho, s.x, s.y, s.z)
    _add_density_term(f, pb, bx, by, bz, s.d1, s.w_bottom, s.w_top, s.dx, s.dy, -1.0)
    return pad(f)


def main():
    folder = sys.argv[1]
    chid = sys.argv[2] if len(sys.argv) > 2 else "flame_periodic"
    if len(sys.argv) <= 2 and not os.path.isfile(os.path.join(folder, chid + ".smv")):
        # If the supplied folder contains a differently named run at its root,
        # use it before considering nested runs (such as the smaller OMP case).
        root_runs = [os.path.splitext(os.path.basename(path))[0]
                     for path in glob.glob(os.path.join(folder, "*.smv"))
                     if glob.glob(os.path.join(folder,
                                               os.path.splitext(os.path.basename(path))[0] + "_*.sf"))]
        if len(root_runs) == 1:
            chid = root_runs[0]
    Lx, Ly, Lz = [float(v) for v in sys.argv[3:6]] if len(sys.argv) > 5 else (1.92, 1.92, 1.92)
    fields, times = load_fields(folder, chid)
    missing = {"DENSITY", "PRESSURE"} - fields.keys()
    if missing:
        available = ", ".join(sorted(fields)) or "none"
        raise ValueError(
            "Could not load the required full-domain DENSITY and PRESSURE slices "
            "for CHID %r from %s (available 3D fields: %s). Check that the "
            "matching full-domain slice files and .smv mesh metadata are present."
            % (chid, folder, available)
        )
    rho_all, p_all = fields["DENSITY"], fields["PRESSURE"]
    # keep the block of consecutive time steps at the end of the record
    dt = np.diff(times)
    start = int(np.nonzero(dt > 1.5 * dt[-1])[0][-1] + 1) if np.any(dt > 1.5 * dt[-1]) else 0
    rho_all, p_all, times = rho_all[start:], p_all[start:], times[start:]
    nt, nx, ny, nz = rho_all.shape
    x, y = np.arange(nx + 1) * Lx / nx, np.arange(ny + 1) * Ly / ny
    z = (np.arange(nz) + 0.5) * Lz / nz
    print("%d consecutive steps, t = %.3f .. %.3f s, grid %d x %d x %d" % (nt, times[0], times[-1], nx, ny, nz), flush=True)

    variants = [("poisson", "picard"), ("poisson", "gmres"), ("stratified", "picard"), ("stratified", "gmres")]
    tols = [1e-6, 1e-10]
    res = {"times": [float(t) for t in times], "grid": [nx, ny, nz], "density_ratio": [], "gamma": [], "Lz_beta": [],
           "lagged_error": [], "step_change": [], "replay_error": [], "runs": {}}
    for op, method in variants:
        for start_kind in ["cold", "warm"]:
            for tol in tols:
                res["runs"]["%s/%s/%s/%g" % (op, method, start_kind, tol)] = {"iterations": [], "error": [], "converged": []}

    plain = FFTFDSolver(x, y, z)
    previous = None
    for n in range(nt):
        rho, p_fds = pad(rho_all[n]), pad(p_all[n])
        f = apply_operator(plain, p_fds, rho)
        p_top = p_fds[:, :, -1]
        wb = plain.w_bottom
        g_bot = wb[0] * p_fds[:, :, 0] + wb[1] * p_fds[:, :, 1] + wb[2] * p_fds[:, :, 2]
        bc = dict(p_top=p_top, dpdz_bottom=g_bot)
        beta = mean_stratification(rho, x, y, z)
        solvers = {"poisson": plain, "stratified": FFTFDSolver(x, y, z, beta=beta)}
        bx, by, bz = log_density_gradient(rho, x, y, z)
        res["density_ratio"].append(float(rho.max() / rho.min()))
        res["gamma"].append(float(2 * Lz / np.pi * np.sqrt(bx**2 + by**2 + bz**2).max()))
        res["Lz_beta"].append(float(Lz * np.abs(beta).max()))
        # reference: tightly converged solution
        ref, info = plain.solve_variable_density(f, rho, method="gmres", tol=1e-13, max_iter=300, **bc)
        nref = np.linalg.norm(ref)
        res["replay_error"].append(float(np.linalg.norm(ref - p_fds) / np.linalg.norm(p_fds)))
        for op, method in variants:
            for start_kind in ["cold", "warm"]:
                p0 = previous if start_kind == "warm" else None
                for tol in tols:
                    p, info = solvers[op].solve_variable_density(f, rho, p0=p0, method=method, tol=tol, max_iter=300, **bc)
                    r = res["runs"]["%s/%s/%s/%g" % (op, method, start_kind, tol)]
                    r["iterations"].append(int(info["iterations"]))
                    r["converged"].append(bool(info["converged"]))
                    r["error"].append(float(np.linalg.norm(p - ref) / nref))
        # one sweep from the previous pressure: the density term lagged by one time step
        if previous is not None:
            p1, _ = plain.solve_variable_density(f, rho, p0=previous, method="picard", tol=0.0, max_iter=1, **bc)
            res["lagged_error"].append(float(np.linalg.norm(p1 - ref) / nref))
            res["step_change"].append(float(np.linalg.norm(previous - ref) / nref))
        previous = ref
        if n % 20 == 0:
            print("step %3d  t=%.3f  rho ratio %.2f  gamma %.1f" % (n, times[n], res["density_ratio"][-1], res["gamma"][-1]), flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "data", "fds_replay.json")
    json.dump(res, open(out, "w"), indent=1)
    print("\ndensity ratio: %.2f .. %.2f   gamma: %.1f .. %.1f   Lz*max|beta|: %.2f .. %.2f" % (
        min(res["density_ratio"]), max(res["density_ratio"]), min(res["gamma"]), max(res["gamma"]), min(res["Lz_beta"]), max(res["Lz_beta"])))
    print("solution vs FDS pressure: max relative difference %.1e" % np.max(res["replay_error"]))
    print("change of the pressure between consecutive steps: median %.2e, max %.2e" % (np.median(res["step_change"]), np.max(res["step_change"])))
    print("error of one lagged sweep (warm): median %.2e, max %.2e" % (np.median(res["lagged_error"]), np.max(res["lagged_error"])))
    print("\n%-34s %8s %8s %8s %10s %5s" % ("operator/method/start/tol", "mean it", "min", "max", "max error", "conv"))
    for key, r in res["runs"].items():
        it = np.array(r["iterations"][1:] if "/warm/" in key else r["iterations"])
        err = np.array(r["error"][1:] if "/warm/" in key else r["error"])
        print("%-34s %8.1f %8d %8d %10.1e %5s" % (key, it.mean(), it.min(), it.max(), err.max(), all(r["converged"])))
    print("written", out)


if __name__ == "__main__":
    main()
