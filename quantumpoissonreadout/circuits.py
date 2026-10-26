"""circuits -- the readout as one Qiskit circuit, for the square inclusion.

Qubits: position (m = 2 log2 N), local (4, l = b0 + 2 b1 + 4 b2 + 8 b3), s_a, s_b, s_c, z, c.

    U_V          inverse QFT on each coordinate, the multiplexor Q(k), QFT
    Q(k)         X on b3 controlled on k = 0; Ry(2 tau(k)) on b1 uniformly controlled by k;
                 Ry on b0 uniformly controlled by (k, b1), 2 omega(k) for b1 = 0, pi/2 for b1 = 1;
                 T on (b0, b1, b2) controlled on b3 = 0; 2n phase gates e^{i pi (k1 + k2) / N}
    U_Sigma^1/2  Ry on its ancilla, controlled -R_chi, Ry^T: block Sigmatilde^{1/2}
    U_VS         U_Sigma^1/2 (s_b) U_V: block Vs, input Pi_N (l = 0, s_b = 0),
                 output Pi_G (b3 = 0, s_b = 0)
    U_f          QSVT of the even f_t on U_VS^dag, real part by an LCU over +phi and -phi on c
    readout      H on position and (b1, b2), U_Sigma^1/2 (s_a), U_f, U_Sigma^1/2 (s_c), H

The all-zero amplitude of the readout is A = g^T f_t(Vs Vs^T) g.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UCRYGate, UnitaryGate

from quantuminversionpoisson.isometry_circuit import T_matrix, exact_angles


def q_mux(N):
    """sum_k |k><k| (x) Q(k) on [position (2n), local (4)], from uniformly controlled rotations."""
    n = int(np.log2(N)); m = 2 * n
    qc = QuantumCircuit(m + 4, name="Q(k)")
    pos, (b0, b1, b2, b3) = list(range(m)), range(m, m + 4)
    tau, om = np.zeros(N * N), np.zeros(N * N)
    for k1 in range(N):
        for k2 in range(N):
            om[k1 * N + k2], tau[k1 * N + k2] = exact_angles(N, k1, k2) if k1 or k2 else (0, 0)
    qc.mcx(pos, b3, ctrl_state=0)                                     # k = 0: |0> -> |8>
    qc.append(UCRYGate(list(2 * tau)), [b1] + pos)
    qc.append(UCRYGate(list(2 * om) + [np.pi / 2] * (N * N)), [b0] + pos + [b1])
    qc.append(UnitaryGate(T_matrix()[:8, :8], label="T").control(1, ctrl_state=0),
              [b3, b0, b1, b2])
    for b in range(n):                                                # e^{i pi (k1 + k2) / N}
        qc.p(np.pi * 2 ** b / N, pos[b]); qc.p(np.pi * 2 ** b / N, pos[n + b])
    return qc


def u_v(N):
    """U_V on [position (2n), local (4)]."""
    n = int(np.log2(N)); m = 2 * n
    qc = QuantumCircuit(m + 4, name="U_V")
    for reg in (range(n), range(n, m)):
        qc.append(QFTGate(n).inverse(), reg)
    qc.append(q_mux(N), range(m + 4))
    for reg in (range(n), range(n, m)):
        qc.append(QFTGate(n), reg)
    return qc


def u_sigma_half(N, rho):
    """On [position (2n), ancilla]: block g0 I + g1 (-R_chi) = Sigmatilde^{1/2} for the square,
    g0, g1 = (1 +- rho^{-1/2}) / 2; chi = (i_hi xor i_lo)(j_hi xor j_lo)."""
    n = int(np.log2(N)); m = 2 * n
    g0, g1 = (1 + rho ** -0.5) / 2, (1 - rho ** -0.5) / 2
    th = 2 * np.arctan2(np.sqrt(g1), np.sqrt(g0))
    jh, jl, ih, il, a = n - 1, n - 2, m - 1, m - 2, m
    qc = QuantumCircuit(m + 1, name="U_Sigma^1/2")
    qc.ry(th, a)
    qc.cx(jl, jh); qc.cx(il, ih)
    qc.ccz(a, ih, jh); qc.z(a)                                        # -R_chi = 2 chi - 1
    qc.cx(il, ih); qc.cx(jl, jh)
    qc.ry(-th, a)
    return qc


def projector_phase(qc, phi, anc, z, c):
    """exp(i phi Z_c (2 Pi - I)), Pi = all of anc in |0>: mark Pi on z, RZZ(2 phi), unmark."""
    qc.mcx(anc, z, ctrl_state=0)
    qc.rzz(2 * phi, c, z)
    qc.mcx(anc, z, ctrl_state=0)


def readout_circuit(N, rho, phi):
    """The readout circuit; returns (circuit, U_V, U_VS)."""
    m = 2 * int(np.log2(N))
    pos, loc = list(range(m)), list(range(m, m + 4))
    sa, sb, sc, z, c = range(m + 4, m + 9)
    UV, Us = u_v(N), u_sigma_half(N, rho)
    UVs = QuantumCircuit(m + 5, name="U_VS")                         # [pos, loc, s_b]
    UVs.append(UV, range(m + 4))
    UVs.append(Us, pos + [m + 4])
    UVsd = UVs.inverse()
    data = pos + loc + [sb]
    PG, PN = [loc[3], sb], loc + [sb]                                # gradient, nodal subspaces
    qc = QuantumCircuit(m + 9, name="readout")
    prep = pos + [loc[1], loc[2]]                                    # ehat_x: local 0, 2, 4, 6
    qc.h(prep)
    qc.append(Us, pos + [sa])                                        # g = Sigmatilde^{1/2} ehat
    qc.h(c)
    d = len(phi) - 1
    for j in range(d, -1, -1):                                       # rightmost factor first
        first = (d - j) % 2 == 0
        projector_phase(qc, phi[j], PG if first else PN, z, c)
        if j > 0:
            qc.append(UVsd if first else UVs, data)
    qc.h(c)
    qc.append(Us, pos + [sc])
    qc.h(prep)
    return qc, UV, UVs


def counts(qc):
    """(qubits, CX, single-qubit gates, depth) after transpiling to {cx, u}, optimization level 1."""
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    ops = t.count_ops()
    return t.num_qubits, ops.get("cx", 0), ops.get("u", 0), t.depth()
