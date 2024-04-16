import numpy as np
import sympy as sym
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

def fftfd_solver(x, y, f, p_top):
    Nx, Ny = x.shape[0], y.shape[0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    F = f[:-1, :-1] # Remove boundary
    kx = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    # For any domain
    #kx = 2 * np.pi * kx / x[-1]
    F_k = np.fft.fft(F, axis=1)
    P_k = np.zeros_like(F_k)
    Dyv = np.zeros(Ny - 1)
    Dyv[1] = 1
    Dyv[-1] = 1
    P_kNy = np.fft.fft(np.ones(Nx - 1) * p_top)
    tmp = []
    for i in range(Nx-1):
        Dyv[0] = -2 - (kx[i] * dy) ** 2
        Dy = spla.circulant(Dyv) / dy ** 2  
        # Fix boundary conditions
        Dy[0, 0] = - 1.5 * dy 
        Dy[0, 1] = 2 * dy
        Dy[0, 2] = - 0.5 * dy
        Dy[0, -1] = 0
        Dy[-1, 0] = 0
        tmp.append(np.min(np.abs(np.linalg.eigvals(Dy))))
        #print(kx[i], ":", np.min(np.abs(np.linalg.eigvals(Dy))))
        F_k[0, i] = 0
        F_k[-1, i] -=  P_kNy[i] / dy ** 2
        P_k[:, i] = np.linalg.solve(Dy, F_k[:, i])
    print(min(tmp))
    P_FFTFD = np.real(np.fft.ifft(P_k, axis=1))
    P_FFTFD = np.vstack([P_FFTFD, np.ones(Nx - 1) * p_top])
    P_FFTFD = np.hstack([P_FFTFD, P_FFTFD[:, 0].reshape(-1, 1)])
    return P_FFTFD

def experiment(Nx, Ny, f, p, p_top, solver):
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    F = f(X, Y)
    P = p(X, Y)
    P_a = solver(x, y, F, p_top)
    return P, P_a

def dd(X):
    D = np.diag(np.abs(X)) # Find diagonal coefficients
    S = np.sum(np.abs(X), axis=1) - D # Find row sum without diagonal
    if np.all(D > S):
        return "Strictly diagonal dominant"
    elif np.all(D >= S):
        return 'Diagonally dominant'
    else:
        return 'NOT diagonally dominant'