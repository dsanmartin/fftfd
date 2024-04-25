import numpy as np
import matplotlib.pyplot as plt

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

def plot(x: np.ndarray, y: np.ndarray, z: np.ndarray, f: np.ndarray) -> None:
    """
    Plot a 3D scalar field using contour plots.

    Parameters
    ----------
    x : np.ndarray (Nx,)
        Array of x-coordinates.
    y : np.ndarray (Ny,)
        Array of y-coordinates.
    z : np.ndarray (Nz,)
        Array of z-coordinates.
    f : np.ndarray (Nx, Ny, Nz)
        3D scalar field.

    Returns
    -------
    None
        This function does not return anything.
    """
    # Get the indices of the middle of the domain to plot
    i, j, k = x.shape[0] // 2, y.shape[0] // 2, z.shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
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
    return None
    