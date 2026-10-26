"""readout -- least-squares readout of the effective conductivity.

    sigma^H_aa / sigma_max = min_w |g_a + Vs w|^2 = g_a^T (I - P) g_a,
    g_a = Sigmatilde^{1/2} ehat_a,   Vs = Sigmatilde^{1/2} V,   P = projector onto range(Vs).

The estimate (1 + e_t) A - e_t sbar = g^T r_t(Vs Vs^T) g is within e_t sbar of sigma^H / sigma_max,
where A = g^T f_t(Vs Vs^T) g >= 0 is the all-zero amplitude of the readout circuit. Three levels:

    matrix-free (every cell, N up to 1024)       sigma_fe, ls_exact, ls_poly
    one Qiskit circuit (square, state vector)    qiskit_readout
    gate-level arithmetic U_V (square)           arithmetic_readout
and the cost of the readout against the readout through the inverse (baseline_cost).
"""
import numpy as np
import scipy.sparse.linalg as spla
from qiskit.quantum_info import Statevector

from quantuminversionpoisson.isometry_circuit import IsometryCircuit

from .cells import Cell, sigma_fe, uv_counts, check_oracle
from .polynomial import poly_plan, r_apply, even_phases, scalar_response, q_poly
from .circuits import readout_circuit, counts
from .emulation import Emulator, exact_unitaries, ae_samples


# ---- matrix-free ---------------------------------------------------------------------------
def operators(cell, a=1):
    """Vs, Vs^T, and g_a for a Cell."""
    sq = np.sqrt(cell.St)
    Vs = lambda w: sq * (cell.A @ cell.Kih(w))
    VsT = lambda y: cell.Kih(cell.A.T @ (sq * y))
    return Vs, VsT, sq * cell.ehat(a)


def ls_exact(cell, a=1):
    """min_w |g_a + Vs w|^2 by LSQR on Vs: the least-squares form, without the cell problem."""
    Vs, VsT, g = operators(cell, a)
    op = spla.LinearOperator((8 * cell.n, cell.n), matvec=Vs, rmatvec=VsT, dtype=float)
    w = spla.lsqr(op, -g, atol=1e-15, btol=1e-15, iter_lim=10_000)[0]
    r = g + Vs(w)
    return float(r @ r)


def ls_poly(cell, eps_p, a=1):
    """(g^T r_t(Vs Vs^T) g, d_f, e_t): the estimate that the circuit delivers without sampling."""
    t, d, e = poly_plan(cell.rho, eps_p)
    Vs, VsT, g = operators(cell, a)
    return float(g @ r_apply(lambda y: Vs(VsT(y)), g, cell.rho, t)), d, e


# ---- one Qiskit circuit ------------------------------------------------------------------------
def qiskit_readout(N, rho=10.0, eps=1e-2):
    """Build the readout circuit for the square, simulate it by state vector, compare with FE."""
    eps_p = eps / 3
    phi, t, f, e = even_phases(rho, eps_p)
    qc, UV, UVs = readout_circuit(N, rho, phi)
    A = Statevector(qc).data[0]
    cell = Cell("square", N, rho)
    s_poly, _, _ = ls_poly(cell, eps_p)
    xs = np.linspace(0, 1, 301)
    nq, cx, u1, dep = counts(qc)
    return dict(N=N, rho=rho, eps=eps, d=2 * t, e_t=e, A=A.real, A_imag=abs(A.imag),
                estimate=(1 + e) * A.real - e * cell.sbar, poly=s_poly, ls=ls_exact(cell),
                fe=sigma_fe(cell), sbar=cell.sbar,
                phase_err=max(abs(scalar_response(phi, x).real - f(x)) for x in xs),
                qubits=nq, cx=cx, oneq=u1, depth=dep, cx_UVs=counts(UVs)[1], cx_UV=counts(UV)[1])


# ---- gate-level arithmetic ---------------------------------------------------------------------
def bits_for(N, d, eps, e_t=0.0):
    """F such that (1 + e_t) d delta <= eps / 3 with the measured delta <= 6 N 2^-F."""
    return int(np.ceil(np.log2(18 * (1 + e_t) * N * d / eps)))


def arithmetic_readout(N, rho=10.0, eps=1e-2, F=None):
    """Readout amplitude of the square with the gate-level U_V and with an exact U_V."""
    eps_p = eps / 3
    phi, t, f, e = even_phases(rho, eps_p)
    F = F or bits_for(N, 2 * t, eps, e)
    Q, chk = IsometryCircuit(n=int(np.log2(N)), F=F).simulate()
    assert chk["work_bits_left_set"] == 0 and chk["inputs_restored"]
    Qe = exact_unitaries(N, Q)
    delta = max(np.linalg.norm(Q[k] - Qe[k], 2) for k in range(N * N))
    Ag = Emulator(N, rho, Q).amplitude(phi).real
    Ae = Emulator(N, rho, Qe).amplitude(phi).real
    cell = Cell("square", N, rho)
    return dict(N=N, F=F, d=2 * t, e_t=e, delta=delta, A_gate=Ag, A_exact=Ae, sbar=cell.sbar,
                bound=cell.sbar * 2 * t * delta, estimate=(1 + e) * Ag - e * cell.sbar,
                fe=sigma_fe(cell))


def amplitude_estimation(A, M, runs, seed):
    """runs estimates of A by canonical amplitude estimation with M evaluation points."""
    return ae_samples(A, M, runs, np.random.default_rng(seed))


# ---- cost at large N -----------------------------------------------------------------------------
def ls_cost(N, rho, eps, sbar, cell="square"):
    """Least-squares readout: (2M - 1) d_f applications of U_V, each with one controlled R_chi."""
    t, d, e = poly_plan(rho, eps / 3)
    F = bits_for(N, d, eps, e)
    M = int(np.ceil(3 * np.pi * (1 + e) / (eps * sbar)))
    q = (2 * M - 1) * d
    uv = uv_counts(N, F)
    return dict(d=d, C=1.0, F=F, M=M, queries_uv=q,
                toffoli=float(q) * (uv["toffoli"] + check_oracle(cell)["toffoli"]),
                qubits=uv["qubits"] + 5)                  # s_a, s_b, s_c, z, c


def baseline_cost(N, rho, eps, sbar, cell="square"):
    """Readout through the inverse: sigma^H = sbar - C A_q, A_q = psi^T q_t(Mtilde) psi / C;
    (2M - 1) d_q queries to U_M, each with two U_V and one controlled R_chi."""
    _, d, C = q_poly(rho, eps / 3)
    F = int(np.ceil(np.log2(36 * N * d * C / eps)))     # 2 d C delta <= eps / 3, delta <= 6 N 2^-F
    M = int(np.ceil(3 * np.pi * C / (eps * sbar)))
    uv = uv_counts(N, F)
    return dict(d=d, C=C, F=F, M=M, queries_uv=2 * (2 * M - 1) * d,
                toffoli=float((2 * M - 1) * d) * (2 * uv["toffoli"] + check_oracle(cell)["toffoli"]),
                qubits=uv["qubits"] + 7)                  # the ancillas of the inversion readout
