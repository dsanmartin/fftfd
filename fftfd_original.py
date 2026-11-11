"""
Original FFT-FD implementation, by Daniel San Martin.

This is the base code developed for the PhD thesis (doi:10.71959/k76w-qn05) and first published in
this repository; the constant-density solver is also in src/pressure.py of
https://github.com/dsanmartin/wildfire. The functions below are copied verbatim from the original
sources and are kept for provenance and comparison:

    thomas_algorithm, fftfd     constant density, uniform grid            (fftfd.py)
    fftfd_non_uniform_z         constant density, non-uniform vertical grid (fftfd.py)
    solve_iterative             variable density, fixed-point iteration   (analysis_density.ipynb)

Status: the three solvers are correct. solver.py reproduces `fftfd` and `fftfd_non_uniform_z` to
round-off, and `solve_iterative` to discretization accuracy (it differences the density term with
np.gradient, one-sided at the lateral ends, where solver.py uses periodic differences).
experiments.py uses `fftfd` as the timing baseline of the paper, and smoke_test.py checks
solver.py against all three.

Limitations of the original code, left as they were: the domain must start at 0 (the wavenumbers
use x[-1] as the domain length); `solve_iterative` requires a uniform vertical grid, and prints
nothing unless log=True; its default tolerance is on the absolute change of p in the infinity norm.

For new work use solver.py, a batched, multithreaded rewrite that adds the non-zero boundary data,
the GMRES iteration and the stratified operator.
"""
import numpy as np
import numba

@numba.jit(nopython=True)
def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Solve a tridiagonal linear system Ax = f using the Thomas algorithm.
    Numba is used to speed up the computation.

    Parameters
    ----------
    a: np.ndarray (N-1,)
        Lower diagonal of the matrix A.
    b: np.ndarray (N,)
        Main diagonal of the matrix A.
    c: np.ndarray (N-1,)
        Upper diagonal of the matrix A.
    f : np.ndarray (N,)
        Right-hand side of the linear system.

    Returns
    -------
    np.ndarray
        Solution of the linear system.

    Notes
    -----
    The Thomas algorithm is a specialized algorithm for solving tridiagonal
    linear systems. It is more efficient than general-purpose algorithms such
    as Gaussian elimination, especially for large systems.
    """
    N = b.shape[0]
    v = np.zeros(N, dtype=np.complex128)
    l = np.zeros(N - 1, dtype=np.complex128)
    y = np.zeros(f.shape, dtype=np.complex128)
    u = np.zeros(f.shape, dtype=np.complex128)
    # Determine L, U
    v[0] = b[0]
    for k in range(1, N):
        l[k-1] = a[k-1] / v[k-1]
        v[k] = b[k] - l[k-1] * c[k-1]
    # Solve Ly = f
    y[0] = f[0]
    for k in range(1, N):
        y[k] = f[k] - l[k-1] * y[k-1]
    # Solve Uu = y
    u[-1] = y[-1] / v[-1]
    # Backward substitution
    for k in range(-1, -N, -1):
        u[k-1] = (y[k-1] - c[k] * u[k]) / v[k-1]
    return u

def fftfd(f: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute the 3D Poisson equation using the FFT2D-FD method.
    FFT2D for x-direction, y-direction and central finite differences for z-direction.

    Parameters
    ----------
    f : array_like
        Input array of shape (Nx, Ny, Nz) containing the right-hand side of the Poisson equation.
    x : array_like
        Array of shape (Nx,) containing the x coordinates.
    y : array_like
        Array of shape (Ny,) containing the y coordinates.
    z : array_like
        Array of shape (Nz,) containing the z coordinates.

    Returns
    -------
    ndarray (Nx, Ny, Nz)
        Approximation of the Poisson equation.
    """
    Nx, Ny, Nz = x.shape[0], y.shape[0], z.shape[0]
    dz = z[1] - z[0]
    x_max = x[-1] # Max x value
    y_max = y[-1] # Max y value
    p_top = np.zeros((Nx, Ny)) # Top boundary condition p = 0
    F = f[:-1, :-1, :-1] # Removing boundaries and top boundary condition
    rr = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    ss = np.fft.fftfreq(Ny - 1) * (Ny - 1)
    # Scale frequencies for any domain
    kx = 2 * np.pi * rr * dz / x_max
    ky = 2 * np.pi * ss * dz / y_max
    # Compute FFT in x direction (column-wise)
    F_k = np.fft.fft2(F, axes=(0, 1))
    P_k = np.zeros_like(F_k)
    # Compute FFT in the last row (top boundary condition)
    P_kNz = np.fft.fft2(p_top, axes=(0, 1))
    # Solve the system for each gamma
    for r in range(Nx - 1):
        for s in range(Ny - 1):
            # Compute gamma
            gamma_rs = - 2 - kx[r] ** 2 - ky[s] ** 2
            # Create RHS of system
            F_k[r, s, 0] = 0 + 0.5 * dz * F_k[r, s, 1] # dp/dy = 0
            # Substract coefficient of top boundary condition
            F_k[r, s, -1] -=  P_kNz[r, s] / dz ** 2 
            # Create A in the system. Only keep diagonals of matrix
            a = np.ones(Nz - 2) / dz ** 2
            b = np.ones(Nz - 1) * gamma_rs / dz ** 2
            c = np.ones(Nz - 2) / dz ** 2
            # Fix first coefficients
            c[0] = (2 + 0.5 * gamma_rs) / dz
            b[0] = -1 / dz
            # Solve system A P_k = F_k
            P_k[r, s, :] = thomas_algorithm(a, b, c, F_k[r, s, :])
    # Compute IFFT in x-y slices to restore the 'physical' pressure
    p = np.real(np.fft.ifft2(P_k, axes=(0, 1)))
    # Add x boundary condition
    p = np.concatenate([p, np.expand_dims(p[0], axis=0)], axis=0)
    # Add y boundary condition
    p = np.concatenate([p, np.expand_dims(p[:, 0], axis=1)], axis=1)
    # Add top boundary condition
    p = np.concatenate([p, np.expand_dims(p_top, axis=2)], axis=2)
    return p

# ---------------------------------------------------------------------------------------
# Non-uniform vertical grid (verbatim from fftfd.py)
# ---------------------------------------------------------------------------------------
def fftfd_non_uniform_z(f: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Compute the 3D Poisson equation using the FFT2D-FD method.
    FFT2D for x-direction, y-direction and central finite differences for z-direction.

    Parameters
    ----------
    f : array_like
        Input array of shape (Nx, Ny, Nz) containing the right-hand side of the Poisson equation.
    x : array_like
        Array of shape (Nx,) containing the x coordinates.
    y : array_like
        Array of shape (Ny,) containing the y coordinates.
    z : array_like
        Array of shape (Nz,) containing the z coordinates.

    Returns
    -------
    ndarray (Nx, Ny, Nz)
        Approximation of the Poisson equation.
    """
    Nx, Ny, Nz = x.shape[0], y.shape[0], z.shape[0]
    dz = z[1:] - z[:-1]
    x_max = x[-1] # Max x value
    y_max = y[-1] # Max y value
    p_top = np.zeros((Nx, Ny)) # Top boundary condition p = 0
    F = f[:-1, :-1, :-1] # Removing boundaries and top boundary condition
    rr = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    ss = np.fft.fftfreq(Ny - 1) * (Ny - 1)
    # Scale frequencies for any domain
    # kx = 2 * np.pi * rr * dz / x_max
    # ky = 2 * np.pi * ss * dz / y_max
    kx = 2 * np.pi * rr / x_max
    ky = 2 * np.pi * ss / y_max
    coef_1 = dz[:-1] + dz[1:] 
    coef_2 = coef_1 * (dz[:-1] * dz[1:])
    # Compute FFT in x direction (column-wise)
    F_k = np.fft.fft2(F, axes=(0, 1))
    P_k = np.zeros_like(F_k)
    # Compute FFT in the last row (top boundary condition)
    P_kNz = np.fft.fft2(p_top, axes=(0, 1))
    b = np.zeros(Nz - 1)   
    c = np.zeros(Nz - 2)
    # Solve the system for each gamma
    for r in range(Nx - 1):
        for s in range(Ny - 1):
            # Create RHS of system
            # F_k[r, s, 0] = 0 + 0.5 * dz * F_k[r, s, 1] # dp/dy = 0
            F_k[r, s, 0] = coef_2[0] * (0 + 0.5 * dz[0] * F_k[r, s, 1]) # dp/dy = 0
            # Substract coefficient of top boundary condition
            F_k[r, s, -1] = coef_2[-1] * F_k[r, s, -1] - 2 * dz[-2] * P_kNz[r, s] 
            # Create A in the system. Only keep diagonals of matrix
            a = 2 * dz[1:]
            c[1:] = 2 * dz[:-2]                    
            b[0] = -dz[1] * coef_1[0] #(dz[0] + dz[1]) ** 2 + 0.5 * dz[0] * b[0]     
            b[1:] = -coef_2 * (kx[r] ** 2 + ky[s] ** 2) - 2 * coef_1
            # c[0] = (dz[0] + dz[1]) ** 2 + 0.5 * dz[0] * b[1] 
            c[0] = (dz[0] + dz[1]) ** 2 - 0.5 * dz[0] * coef_2[0] * (kx[r] ** 2 + ky[s] ** 2) - dz[0] * coef_1[0]                             
            ff = F_k[r, s, :]
            ff[1:-1] = coef_2[:-1] * ff[1:-1]
            # print(z.shape, dz.shape, a.shape, b.shape, c.shape, ff.shape, coef_1.shape, coef_2.shape)  
            # Solve system A P_k = F_k
            P_k[r, s, :] = thomas_algorithm(a, b, c, ff)
    # Compute IFFT in x-y slices to restore the 'physical' pressure
    p = np.real(np.fft.ifft2(P_k, axes=(0, 1)))
    # Add x boundary condition
    p = np.concatenate([p, np.expand_dims(p[0], axis=0)], axis=0)
    # Add y boundary condition
    p = np.concatenate([p, np.expand_dims(p[:, 0], axis=1)], axis=1)
    # Add top boundary condition
    p = np.concatenate([p, np.expand_dims(p_top, axis=2)], axis=2)
    return p

# ---------------------------------------------------------------------------------------
# Variable density: fixed-point iteration (verbatim from analysis_density.ipynb)
# ---------------------------------------------------------------------------------------
def solve_iterative(x, y, z, p0, rho, f, tol=1e-14, max_iter=20, log=False):
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    for m in range(1, max_iter+1):
        px = np.gradient(p0, dx, axis=0, edge_order=2)
        py = np.gradient(p0, dy, axis=1, edge_order=2)
        pz = np.gradient(p0, dz, axis=2, edge_order=2)
        rhox = np.gradient(rho, dx, axis=0, edge_order=2)
        rhoy = np.gradient(rho, dy, axis=1, edge_order=2)
        rhoz = np.gradient(rho, dz, axis=2, edge_order=2)
        f_new = f + (rhox * px + rhoy * py + rhoz * pz) / rho 
        p = fftfd(f_new, x, y, z)
        error = np.linalg.norm(p.flatten() - p0.flatten(), np.inf)
        p0 = p.copy()
        if log:
            print("m = %d, error = %.e" % (m, error))
        if error < tol:        
            break
    return p
