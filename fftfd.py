import numpy as np
#import sympy as sym
import scipy.linalg as spla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import matplotlib.pyplot as plt


def plot(x, y, z, f):
    i, j, k = x.shape[0] // 2, y.shape[0] // 2, z.shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    # c1 = ax[0].contourf(x, y, f[:, :, k])
    # c2 = ax[1].contourf(x, z, f[j, :, :].T)
    # c3 = ax[2].contourf(y, z, f[:, i, :].T)
    c1 = ax[0].contourf(x, y, f[:, :, k].T)
    c2 = ax[1].contourf(x, z, f[:, j, :].T)
    c3 = ax[2].contourf(y, z, f[i, :, :].T)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$z$')
    ax[2].set_xlabel(r'$y$')
    ax[2].set_ylabel(r'$z$')
    fig.colorbar(c1, ax=ax[0])
    fig.colorbar(c2, ax=ax[1])
    fig.colorbar(c3, ax=ax[2])
    plt.tight_layout()
    plt.show()
    
def error(P, P_a, norm=np.inf):
    return np.linalg.norm(P - P_a, norm) / np.linalg.norm(P, norm)

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
    # a, b, c = A # Get diagonals
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
    #for k in range(N-1, -1, -1):
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


def poisson_iterative(x, y, p, f, p_top, tol=1e-10, n_iter=100):
    dx, dy = x[1] - x[0], y[1] - y[0]
    for n in range(n_iter):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            dy ** 2 * (p[1:-1, 2:] + p[1:-1, :-2]) + 
            dx ** 2 * (p[2:, 1:-1] + p[:-2, 1:-1]) - 
            dx **2 * dy ** 2 * f[1:-1, 1:-1]
        ) / (2 * (dx ** 2 + dy ** 2))
        # Boundary conditions
        # x periodic
        p[1:-1, 0] = (
            dy ** 2 * (p[1:-1, 1] + p[1:-1, -1]) + 
            dx ** 2 * (p[2:, 0] + p[:-2, 0]) - 
            dx **2 * dy ** 2 * f[1:-1, 0]
        ) / (2 * (dx ** 2 + dy ** 2))
        p[1:-1, -1] = (
            dy ** 2 * (p[1:-1, 0] + p[1:-1, -2]) + 
            dx ** 2 * (p[2:, -1] + p[:-2, -1]) - 
            dx **2 * dy ** 2 * f[1:-1, -1]
        ) / (2 * (dx ** 2 + dy ** 2))
        # y
        p[0] = (4 * p[1] - p[2]) / 3 # dp/dy = 0 at y=y_min
        p[-1] = p_top # p = p_top at y=y_max
        if error(p, pn) < tol:
            break
    return p

def compute_matrix_vector_product(x, Dx, Dy):
    Nx, Ny = Dx.shape[0], Dy.shape[0]
    X = np.reshape(x, (Ny, Nx))
    out = Dy.dot(X) + X.dot(Dx)
    return out.flatten()

def fd_solver(x, y, f, p_top, method='gmres', tol=1e-10, n_iter=1000):
    Nx, Ny = x.shape[0], y.shape[0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    Dxv = np.zeros(Nx - 1)
    Dxv[0] = -2
    Dxv[1] = 1
    Dxv[-1] = 1
    Dx = spla.circulant(Dxv) / dx ** 2
    Dyv = np.zeros(Ny - 2)
    Dyv[0] = -2
    Dyv[1] = 1
    Dyv[-1] = 1
    Dy = spla.circulant(Dyv) 
    # BC 
    Dy[0, 0] = -2 / 3
    Dy[0, 1] = 1 / 3
    Dy[-1, 0] = 0
    Dy = Dy / dy ** 2
    #print(Dy * dy ** 2)
    Ix = np.eye(Nx - 1)
    Iy = np.eye(Ny - 2)
    F_ = f[1:-1, :-1]
    #F_[0] = 0 
    F_[-1] -= p_top / dy ** 2
    F = F_.flatten()
    #print(Dx.shape, Dy.shape, F_.shape, F.shape)
    #return False
    if method == 'gmres': # Solving using GMRES
        Av = lambda v: compute_matrix_vector_product(v, Dx, Dy)
        afun = spspla.LinearOperator(shape=((Ny - 2) ** 2, (Nx - 1) ** 2), matvec=Av)
        P_a, _ = spspla.gmres(afun, F, tol=tol, maxiter=n_iter)
    elif method == 'iter': # Solving using Jacobi
        F_ = f[:, :-1]
        p0 = np.zeros_like(F_)
        P_a = poisson_iterative(x, y, p0, F_, p_top, tol=tol, n_iter=n_iter)
        P_a = P_a.reshape(Ny, Nx - 1)
        P_a = np.hstack([P_a, P_a[:, 0].reshape(-1, 1)])
        return P_a
    elif method == 'direct': # PALU
        L = np.kron(Ix, Dx) + np.kron(Dy, Iy)
        P_a = np.linalg.solve(L, F)
    P_a = P_a.reshape(Ny - 2, Nx - 1)
    PP = np.zeros_like(f)
    P_0 = (4 * P_a[1] - P_a[2]) / 3
    PP[0, :-1] = P_0
    PP[1:-1, :-1] = P_a
    PP[-1] = p_top
    PP[:,-1] = PP[:,0]
    #P_a = np.vstack([P_0.reshape(1, -1), P_a])
    #P_a = np.vstack([P_a, np.ones(Ny - 1) * p_top])
    #P_a = np.hstack([P_a, P_a[:, 0].reshape(-1, 1)])
    return PP#P_a


def dd(X):
    D = np.diag(np.abs(X)) # Find diagonal coefficients
    S = np.sum(np.abs(X), axis=1) - D # Find row sum without diagonal
    if np.all(D > S):
        return "Strictly diagonal dominant"
    elif np.all(D >= S):
        return 'Diagonally dominant'
    else:
        return 'NOT diagonally dominant'