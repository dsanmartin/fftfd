"""
Manufactured solutions for  lap(p) - grad(log rho) . grad(p) = f  on [0, Lx] x [0, Ly] x [0, Lz],
periodic in x and y, dp/dz = 0 at z = 0 and p = 0 at z = Lz.

Every field is a product  Z(z) * H(x, y)  so all derivatives are available in closed form.
"""
import numpy as np


class VonMises:
    """H(x, y) = exp(a (cos(wx (x - x0)) + cos(wy (y - y0)) - 2)): exactly periodic bump."""

    def __init__(self, Lx, Ly, a=2.0, x0=None, y0=None):
        self.wx, self.wy, self.a = 2 * np.pi / Lx, 2 * np.pi / Ly, a
        self.x0 = Lx / 2 if x0 is None else x0
        self.y0 = Ly / 2 if y0 is None else y0

    def __call__(self, x, y):
        cx, sx = np.cos(self.wx * (x - self.x0)), np.sin(self.wx * (x - self.x0))
        cy, sy = np.cos(self.wy * (y - self.y0)), np.sin(self.wy * (y - self.y0))
        a, wx, wy = self.a, self.wx, self.wy
        H = np.exp(a * (cx + cy - 2))
        Hx, Hy = -a * wx * sx * H, -a * wy * sy * H
        Hxx = (a**2 * sx**2 - a * cx) * wx**2 * H
        Hyy = (a**2 * sy**2 - a * cy) * wy**2 * H
        return H, Hx, Hy, Hxx, Hyy


class Gaussian2D:
    """H(x, y) = exp(-((x - x0)^2 / sx + (y - y0)^2 / sy)): the thesis test (periodic up to ~1e-9)."""

    def __init__(self, x0, y0, sx, sy):
        self.x0, self.y0, self.sx, self.sy = x0, y0, sx, sy

    def __call__(self, x, y):
        H = np.exp(-((x - self.x0) ** 2 / self.sx + (y - self.y0) ** 2 / self.sy))
        Hx, Hy = -2 * (x - self.x0) / self.sx * H, -2 * (y - self.y0) / self.sy * H
        Hxx = (4 * (x - self.x0) ** 2 / self.sx**2 - 2 / self.sx) * H
        Hyy = (4 * (y - self.y0) ** 2 / self.sy**2 - 2 / self.sy) * H
        return H, Hx, Hy, Hxx, Hyy


def z_thesis(z, z0=np.pi, sz=0.5):
    """Z = (2 pi z^2 - z^3) / pi^3 * exp(-(z - z0)^2 / sz) on [0, 2 pi]."""
    g, gz, gzz = (2 * np.pi * z**2 - z**3) / np.pi**3, (4 * np.pi * z - 3 * z**2) / np.pi**3, (4 * np.pi - 6 * z) / np.pi**3
    E = np.exp(-((z - z0) ** 2) / sz)
    Ez = -2 * (z - z0) / sz * E
    Ezz = (4 * (z - z0) ** 2 / sz**2 - 2 / sz) * E
    return g * E, gz * E + g * Ez, gzz * E + 2 * gz * Ez + g * Ezz


def z_layer(z, Lz, delta):
    """Z = exp(-(z / delta)^2) - exp(-(Lz / delta)^2): layer of thickness delta attached to the bottom."""
    E = np.exp(-((z / delta) ** 2))
    return E - np.exp(-((Lz / delta) ** 2)), -2 * z / delta**2 * E, (4 * z**2 / delta**4 - 2 / delta**2) * E


def fields(x, y, z, Zp, Hp, theta=0.0, Zr=None, Hr=None, strat=0.0):
    """
    Exact p, density rho and source f on the grid.

    p   = Zp(z) Hp(x, y)
    rho = exp(-strat z) / (1 + theta Zr(z) Hr(x, y))
          The second factor is an ideal gas at constant pressure with T / T_inf = 1 + theta Zr Hr
          (density ratio 1 + theta); the first one is an exponentially stratified background
          with scale height 1 / strat. theta = 0 and strat = 0 give constant density.
    """
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Zv, Zz, Zzz = Zp(Z)
    H, Hx, Hy, Hxx, Hyy = Hp(X, Y)
    p = Zv * H
    f = Zv * (Hxx + Hyy) + Zzz * H
    rho = np.ones_like(p)
    if theta != 0.0:
        Rv, Rz, _ = Zr(Z)
        Q, Qx, Qy, _, _ = Hr(X, Y)
        tau = 1 + theta * Rv * Q  # T / T_inf
        # grad(log rho) = -grad(tau) / tau
        bx, by, bz = -theta * Rv * Qx / tau, -theta * Rv * Qy / tau, -theta * Rz * Q / tau
        f = f - (bx * Zv * Hx + by * Zv * Hy + bz * Zz * H)
        rho = 1 / tau
    if strat != 0.0:
        f = f + strat * Zz * H
        rho = rho * np.exp(-strat * Z)
    return p, rho, f
