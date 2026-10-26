"""polynomial -- the range-projection polynomial and its QSVT phases.

    r_t(y) = T_t((1 + a - 2 y) / (1 - a)) / T_t(z0),  a = 1 / rho,  z0 = (1 + a) / (1 - a),
    e_t = 1 / T_t(z0),   f_t(x) = (r_t(x^2) + e_t) / (1 + e_t),   degree d_f = 2 t.

f_t is even, f_t(0) = 1, 0 <= f_t <= 1 on [-1, 1], and 0 <= f_t <= 2 e_t / (1 + e_t) on
[rho^{-1/2}, 1]. Also the odd inverse polynomial q_t of the inversion paper, for the baseline.
"""
import contextlib
import io

import numpy as np
from numpy.polynomial import chebyshev as Cheb


def poly_plan(rho, eps_p):
    """(t, d_f = 2 t, e_t) with t the smallest degree such that e_t <= eps_p."""
    z0 = (1 + 1 / rho) / (1 - 1 / rho)
    t = int(np.ceil(np.arccosh(1 / eps_p) / np.arccosh(z0)))
    return t, 2 * t, 1 / np.cosh(t * np.arccosh(z0))


def f_poly(rho, t):
    """f_t as a function of x on [-1, 1]."""
    a = 1 / rho
    z0 = (1 + a) / (1 - a)
    e = 1 / np.cosh(t * np.arccosh(z0))

    def f(x):
        z = (1 + a - 2 * np.asarray(x, float) ** 2) / (1 - a)
        Tz = np.where(np.abs(z) <= 1, np.cos(t * np.arccos(np.clip(z, -1, 1))),
                      np.cosh(t * np.arccosh(np.maximum(np.abs(z), 1))))
        return (Tz * e + e) / (1 + e)
    return f


def r_apply(Y, g, rho, t):
    """r_t(Y) g by the Chebyshev three-term recurrence: t applications of Y."""
    a = 1 / rho
    z0 = (1 + a) / (1 - a)
    Z = lambda v: ((1 + a) * v - 2 * Y(v)) / (1 - a)
    T0, T1 = g, Z(g)
    for _ in range(t - 1):
        T0, T1 = T1, 2 * Z(T1) - T0
    return T1 / np.cosh(t * np.arccosh(z0))


def even_phases(rho, eps_p):
    """Reflection-convention QSVT phases with Re <0|U(x)|0> = f_t(x); returns (phi, t, f, e_t).
    Symmetric-QSP Newton solver of pyqsp (Dong et al.), mapped from the Wx convention."""
    from pyqsp.sym_qsp_opt import newton_solver
    t, d, e = poly_plan(rho, eps_p)
    f = f_poly(rho, t)
    c = Cheb.chebinterpolate(f, d)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, _, seq = newton_solver(c[0::2], 0, crit=1e-13, maxiter=60)
    ph = np.array(seq.full_phases, float)
    ph[0] -= np.pi / 2                                    # Wx convention of the solver
    phi = ph - np.pi / 2                                  # Wx -> reflection convention
    phi[0] += np.pi / 4 + d * np.pi / 2
    phi[-1] += np.pi / 4
    return phi, t, f, e


def scalar_response(phi, x):
    """<0| e^{i phi_0 Z} R(x) e^{i phi_1 Z} ... R(x) e^{i phi_d Z} |0>, R(x) the reflection."""
    s = np.sqrt(1 - x * x)
    U = np.diag(np.exp(1j * phi[0] * np.array([1, -1])))
    for p in phi[1:]:
        U = U @ np.array([[x, s], [s, -x]]) @ np.diag(np.exp(1j * p * np.array([1, -1])))
    return U[0, 0]


def q_poly(rho, eps_p):
    """Odd q_t of the inversion paper, x q_t(x) = 1 - r_t(x^2) on [1/rho^2, 1]; (q, d_q, C)."""
    a = 1 / rho ** 2
    s = (1 + a) / (1 - a)
    t = int(np.ceil(0.5 * rho * np.log(2 / eps_p)))
    Ts = np.cosh(t * np.arccosh(s))

    def q(x):
        z = (1 + a - 2 * x ** 2) / (1 - a)
        Tz = np.where(np.abs(z) <= 1, np.cos(t * np.arccos(np.clip(z, -1, 1))),
                      np.cosh(t * np.arccosh(np.maximum(np.abs(z), 1))))
        return (1 - Tz / Ts) / x
    C = float(np.abs(q(np.linspace(1e-9, 1.0, 2_000_001))).max())
    return q, 2 * t - 1, C
