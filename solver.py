"""
FFT-FD pressure solver.

Solves, on a box that is periodic in x and y,

    lap(p) - grad(log rho) . grad(p) = f,     dp/dz = 0 at z = z_min,   p = p_top at z = z_max,

with a 2D FFT in the horizontal directions and second-order finite differences on a
(possibly non-uniform) vertical grid.

* ``FFTFDSolver.solve``                  constant-density Poisson problem (direct).
* ``FFTFDSolver(..., beta=...)``         density that depends on z only (direct).
* ``FFTFDSolver.solve_variable_density`` general rho(x, y, z): fixed-point (Picard) or
                                         GMRES iteration preconditioned by the direct solver.

Grid convention: ``x`` and ``y`` contain both end points, so the last
node is the periodic image of the first one and there are ``Nx - 1`` independent nodes.
"""
import time

import numba
import numpy as np
import scipy.fft as sfft
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres


# --------------------------------------------------------------------------------------
# Finite-difference weights on a non-uniform grid
# --------------------------------------------------------------------------------------
def fd_weights(z: np.ndarray):
    """
    Three-point weights for the first and second derivatives at the interior nodes
    z_1, ..., z_{N-2} of a non-uniform grid, and one-sided first-derivative weights at
    both ends. Each returned array has shape (3, N) (columns 0 and N-1 of the centred
    weights are unused): ``d[0, k] * q[k-1] + d[1, k] * q[k] + d[2, k] * q[k+1]``.
    """
    N = z.shape[0]
    h = np.diff(z)
    hm, hp = h[:-1], h[1:]  # h_{k-1}, h_k for k = 1..N-2
    d1 = np.zeros((3, N))
    d2 = np.zeros((3, N))
    d1[0, 1:-1] = -hp / (hm * (hm + hp))
    d1[1, 1:-1] = (hp - hm) / (hm * hp)
    d1[2, 1:-1] = hm / (hp * (hm + hp))
    d2[0, 1:-1] = 2 / (hm * (hm + hp))
    d2[1, 1:-1] = -2 / (hm * hp)
    d2[2, 1:-1] = 2 / (hp * (hm + hp))
    # One-sided, second order: bottom uses nodes 0, 1, 2 and top uses N-3, N-2, N-1
    h0, h1 = h[0], h[1]
    bottom = np.array([-(2 * h0 + h1) / (h0 * (h0 + h1)), (h0 + h1) / (h0 * h1), -h0 / (h1 * (h0 + h1))])
    hb, ha = h[-1], h[-2]
    top = np.array([hb / (ha * (ha + hb)), -(ha + hb) / (ha * hb), (ha + 2 * hb) / (hb * (ha + hb))])
    return d1, d2, bottom, top


# --------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _solve_modes(lower, diag, upper, kappa, r0_diag, r0_diag_k, r0_up, r0_up_k, rhs):
    """
    Thomas algorithm for every horizontal mode, in place.

    Row k >= 1 of mode m is  lower[k] q[k-1] + (diag[k] - kappa[m]) q[k] + upper[k] q[k+1].
    Row 0 is                 (r0_diag + r0_diag_k kappa[m]) q[0] + (r0_up + r0_up_k kappa[m]) q[1].
    ``rhs`` has shape (M, n); the modes are independent, so the loop over m is parallel.
    """
    M, n = rhs.shape
    for m in prange(M):
        kap = kappa[m]
        cp = np.empty(n)
        b = r0_diag + r0_diag_k * kap
        cp[0] = (r0_up + r0_up_k * kap) / b
        rhs[m, 0] = rhs[m, 0] / b
        for k in range(1, n):
            den = diag[k] - kap - lower[k] * cp[k - 1]
            cp[k] = upper[k] / den
            rhs[m, k] = (rhs[m, k] - lower[k] * rhs[m, k - 1]) / den
        for k in range(n - 2, -1, -1):
            rhs[m, k] = rhs[m, k] - cp[k] * rhs[m, k + 1]


@njit(parallel=True, cache=True)
def _add_density_term(g, p, bx, by, bz, d1, bottom, top, dx, dy, sign):
    """
    g += sign * (bx dp/dx + by dp/dy + bz dp/dz) on the periodic block (nx, ny, nz).
    Central differences (periodic in x, y; three-point non-uniform in z; one-sided at the ends).
    """
    nx, ny, nz = p.shape
    for i in prange(nx):
        ip, im = (i + 1) % nx, (i - 1) % nx
        for j in range(ny):
            jp, jm = (j + 1) % ny, (j - 1) % ny
            for k in range(nz):
                if k == 0:
                    pz = bottom[0] * p[i, j, 0] + bottom[1] * p[i, j, 1] + bottom[2] * p[i, j, 2]
                elif k == nz - 1:
                    pz = top[0] * p[i, j, k - 2] + top[1] * p[i, j, k - 1] + top[2] * p[i, j, k]
                else:
                    pz = d1[0, k] * p[i, j, k - 1] + d1[1, k] * p[i, j, k] + d1[2, k] * p[i, j, k + 1]
                px = (p[ip, j, k] - p[im, j, k]) / (2 * dx)
                py = (p[i, jp, k] - p[i, jm, k]) / (2 * dy)
                g[i, j, k] += sign * (bx[i, j, k] * px + by[i, j, k] * py + bz[i, j, k] * pz)


def log_density_gradient(rho, x, y, z):
    """grad(rho) / rho on the periodic block, with the same differences used for p."""
    r = np.ascontiguousarray(rho[:-1, :-1, :])
    dx, dy = x[1] - x[0], y[1] - y[0]
    d1, _, bottom, top = fd_weights(z)
    bx = (np.roll(r, -1, 0) - np.roll(r, 1, 0)) / (2 * dx) / r
    by = (np.roll(r, -1, 1) - np.roll(r, 1, 1)) / (2 * dy) / r
    rz = np.empty_like(r)
    rz[:, :, 1:-1] = d1[0, 1:-1] * r[:, :, :-2] + d1[1, 1:-1] * r[:, :, 1:-1] + d1[2, 1:-1] * r[:, :, 2:]
    rz[:, :, 0] = bottom[0] * r[:, :, 0] + bottom[1] * r[:, :, 1] + bottom[2] * r[:, :, 2]
    rz[:, :, -1] = top[0] * r[:, :, -3] + top[1] * r[:, :, -2] + top[2] * r[:, :, -1]
    return bx, by, rz / r


# --------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------
class FFTFDSolver:
    """
    Direct FFT-FD solver for  lap(p) - beta(z) dp/dz = f  (beta = 0 gives the Poisson problem).

    Parameters
    ----------
    x, y : uniform coordinates including both end points (periodic directions).
    z : vertical coordinates, uniform or not, from the bottom (Neumann) to the top (Dirichlet).
    beta : optional array (Nz,), d(log rho_bar)/dz of a density profile that depends on z only.
    bottom : 'one-sided' (second-order one-sided difference, as in the original implementation) or
             'ghost' (mirror node; uses the equation at z_0 and keeps the matrix diagonally dominant).
    workers : threads used by the FFTs (the tridiagonal solves use numba's thread pool).
    """

    def __init__(self, x, y, z, beta=None, bottom="one-sided", workers=-1):
        self.x, self.y, self.z = x, y, z
        self.Nx, self.Ny, self.Nz = x.shape[0], y.shape[0], z.shape[0]
        self.nx, self.ny, self.n = self.Nx - 1, self.Ny - 1, self.Nz - 1
        self.dx, self.dy = x[1] - x[0], y[1] - y[0]
        self.workers = workers
        self.bottom_closure = bottom
        Lx, Ly = x[-1] - x[0], y[-1] - y[0]
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=1.0 / self.nx) / Lx
        ky = 2 * np.pi * np.fft.rfftfreq(self.ny, d=1.0 / self.ny) / Ly
        self.kappa2d = kx[:, None] ** 2 + ky[None, :] ** 2
        self.kappa = np.ascontiguousarray(self.kappa2d.ravel())
        self.d1, self.d2, self.w_bottom, self.w_top = fd_weights(z)
        self.beta = np.zeros(self.Nz) if beta is None else np.asarray(beta, dtype=float)
        # Rows k = 1..Nz-2:  lower q_{k-1} + (diag - kappa) q_k + upper q_{k+1} = f_k
        n = self.n
        self.lower, self.diag, self.upper = np.zeros(n), np.zeros(n), np.zeros(n)
        self.lower[1:] = self.d2[0, 1:-1] - self.beta[1:-1] * self.d1[0, 1:-1]
        self.diag[1:] = self.d2[1, 1:-1] - self.beta[1:-1] * self.d1[1, 1:-1]
        self.upper[1:] = self.d2[2, 1:-1] - self.beta[1:-1] * self.d1[2, 1:-1]
        self.top_coupling = self.upper[-1]  # multiplies p_top in the last row
        if bottom == "one-sided":
            # dp/dz(z_0) = 0 with nodes 0, 1, 2; q_2 is eliminated with the equation at z_1
            w0, w1, w2 = self.w_bottom
            e = w2 / self.upper[1]
            self.r0 = (w0 - e * self.lower[1], 0.0, w1 - e * self.diag[1], e)
            self.r0_rhs = ("f1", -e)
        elif bottom == "ghost":
            # mirror node q_{-1} = q_1 in the equation at z_0 (the beta term vanishes there)
            h0 = z[1] - z[0]
            self.r0 = (-2 / h0**2, -1.0, 2 / h0**2, 0.0)
            self.r0_rhs = ("f0", 1.0)
        else:
            raise ValueError("bottom must be 'one-sided' or 'ghost'")
        self.timings = {"fft": 0.0, "tridiagonal": 0.0, "rhs": 0.0, "solves": 0}

    # -- helpers ------------------------------------------------------------------------
    def _pad(self, p):
        """Add the periodic images in x and y."""
        out = np.empty((self.Nx, self.Ny, self.Nz))
        out[:-1, :-1] = p
        out[-1, :-1] = p[0]
        out[:, -1] = out[:, 0]
        return out

    def _solve_block(self, g, p_top=None, dpdz=None):
        """Solve on the periodic block. g: (nx, ny, Nz) real. Returns (nx, ny, Nz)."""
        n = self.n
        t0 = time.perf_counter()
        G = sfft.rfft2(g[:, :, :n], axes=(0, 1), workers=self.workers)
        t1 = time.perf_counter()
        which, scale = self.r0_rhs
        G[:, :, 0] = scale * (G[:, :, 1] if which == "f1" else G[:, :, 0])
        if dpdz is not None:  # inhomogeneous Neumann data dp/dz = dpdz at the bottom
            c = 1.0 if which == "f1" else 2.0 / (self.z[1] - self.z[0])
            G[:, :, 0] += c * sfft.rfft2(dpdz, workers=self.workers)
        if p_top is not None:
            G[:, :, -1] -= self.top_coupling * sfft.rfft2(p_top, workers=self.workers)
        shape = G.shape
        G = np.ascontiguousarray(G).reshape(-1, n)
        _solve_modes(self.lower, self.diag, self.upper, self.kappa, *self.r0, G)
        t2 = time.perf_counter()
        p = np.empty((self.nx, self.ny, self.Nz))
        p[:, :, :n] = sfft.irfft2(G.reshape(shape), s=(self.nx, self.ny), axes=(0, 1), workers=self.workers)
        p[:, :, n] = 0.0 if p_top is None else p_top
        t3 = time.perf_counter()
        self.timings["fft"] += (t1 - t0) + (t3 - t2)
        self.timings["tridiagonal"] += t2 - t1
        self.timings["solves"] += 1
        return p

    # -- public API ---------------------------------------------------------------------
    def solve(self, f, p_top=None, dpdz_bottom=None):
        """
        Direct solve. f: (Nx, Ny, Nz); p_top: optional (Nx, Ny) Dirichlet data at the top;
        dpdz_bottom: optional (Nx, Ny) Neumann data dp/dz at the bottom (default 0).
        """
        pt = None if p_top is None else np.ascontiguousarray(p_top[:-1, :-1])
        gb = None if dpdz_bottom is None else np.ascontiguousarray(dpdz_bottom[:-1, :-1])
        return self._pad(self._solve_block(f[:-1, :-1, :], pt, gb))

    def solve_variable_density(self, f, rho, p0=None, p_top=None, method="picard", tol=1e-10,
                               max_iter=200, omega=1.0, restart=30, callback=None, dpdz_bottom=None):
        """
        Solve  lap(p) - grad(rho)/rho . grad(p) = f  for a general density rho(x, y, z).

        The part of grad(log rho) already included in the direct operator (``beta``) is removed
        from the explicit term, so ``beta = 0`` gives the plain Poisson preconditioner and
        ``beta = d(mean log rho)/dz`` gives the stratified one.

        method : 'picard'   p^{m+1} = (1 - omega) p^m + omega L^{-1}(f + B p^m)
                 'gmres'    GMRES on (I - L^{-1} B) p = L^{-1} f
        Returns (p, info). info['residuals'] holds the stopping quantity at each iteration:
        |p^{m+1} - p^m|_inf / |p^{m+1}|_inf for Picard, the relative preconditioned residual for
        GMRES. info['pr_norm'] is the relative preconditioned residual in the 2-norm for both
        (for Picard it is normalised by the first one, which equals GMRES's when p0 = 0).
        """
        fb = np.ascontiguousarray(f[:-1, :-1, :])
        bx, by, bz = log_density_gradient(rho, self.x, self.y, self.z)
        bz = bz - self.beta[None, None, :]
        pt = None if p_top is None else np.ascontiguousarray(p_top[:-1, :-1])
        gb = None if dpdz_bottom is None else np.ascontiguousarray(dpdz_bottom[:-1, :-1])
        args = (bx, by, bz, self.d1, self.w_bottom, self.w_top, self.dx, self.dy, 1.0)

        def T(p):  # one sweep: L^{-1}(f + B p)
            t0 = time.perf_counter()
            g = fb.copy()
            _add_density_term(g, p, *args)
            self.timings["rhs"] += time.perf_counter() - t0
            return self._solve_block(g, pt, gb)

        p = np.zeros_like(fb) if p0 is None else np.ascontiguousarray(p0[:-1, :-1, :]).copy()
        if pt is not None:
            p[:, :, -1] = pt
        res = []
        if method == "picard":
            converged, pr, pr0 = False, [], None
            for m in range(1, max_iter + 1):
                p_new = T(p)
                step = np.linalg.norm(p_new - p)  # |T(p) - p|_2, the preconditioned residual
                pr0 = step if pr0 is None else pr0
                pr.append(step / pr0)
                if omega != 1.0:
                    p_new = (1 - omega) * p + omega * p_new
                diff = np.abs(p_new - p).max() / max(np.abs(p_new).max(), 1e-300)
                p = p_new
                res.append(diff)
                if callback is not None:
                    callback(m, self._pad(p))
                if not np.isfinite(diff) or diff > 1e12:
                    break
                if diff < tol:
                    converged = True
                    break
            return self._pad(p), {"iterations": len(res), "residuals": np.array(res), "converged": converged,
                                  "pr_norm": np.array(pr)}

        # Krylov: unknowns are the nodes below the top boundary
        n, shape = self.n, (self.nx, self.ny, self.n)

        def full(u):
            q = np.empty((self.nx, self.ny, self.Nz))
            q[:, :, :n] = u.reshape(shape)
            q[:, :, n] = 0.0 if pt is None else pt
            return q

        T0 = T(full(np.zeros(self.nx * self.ny * n)))[:, :, :n].ravel()  # affine part

        def matvec(u):  # (I - L^{-1} B) u, linear part of u - T(u)
            return u - (T(full(u))[:, :, :n].ravel() - T0)

        A = LinearOperator((T0.size, T0.size), matvec=matvec, dtype=float)
        u0 = p[:, :, :n].ravel()
        if method != "gmres":
            raise ValueError("method must be 'picard' or 'gmres'")
        u, flag = gmres(A, T0, x0=u0, rtol=tol, atol=0.0, restart=restart,
                        maxiter=max(1, max_iter // restart), callback=lambda r: res.append(r),
                        callback_type="pr_norm")
        return self._pad(full(u)), {"iterations": len(res), "residuals": np.array(res), "converged": flag == 0,
                                    "pr_norm": np.array(res)}


def mean_stratification(rho, x, y, z):
    """Horizontal mean of d(log rho)/dz, the ``beta`` of the stratified direct solver."""
    return log_density_gradient(rho, x, y, z)[2].mean(axis=(0, 1))


def stretched_grid(z_min, z_max, Nz, k):
    """Exponential stretching that clusters nodes near z_min (k = 0 gives a uniform grid)."""
    xi = np.linspace(0.0, 1.0, Nz)
    if k == 0:
        return z_min + (z_max - z_min) * xi
    return z_min + (z_max - z_min) * np.expm1(k * xi) / np.expm1(k)


def set_threads(n):
    """Number of threads used by the tridiagonal and right-hand-side kernels."""
    numba.set_num_threads(n)
