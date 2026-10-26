# QuantumPoissonReadout

Least-squares quantum readout of the effective conductivity of periodic two-phase heat-conduction
cells. The code reproduces every table and figure of the paper

> K. Suresh, *Quantum Readout of the Effective Conductivity of Periodic Two-Phase Media by a
> Least-Squares Range Projection*.

It builds on the block encodings of the companion paper and its package
[QuantumInversionPoisson](https://github.com/UW-ERSL/QuantumInversionPoisson): the partial isometry
`V`, its circuit `U_V`, and the gate-level fixed-point arithmetic of the wavenumber-controlled stage
([MultiplexedStatePreparation](https://github.com/UW-ERSL/MultiplexedStatePreparation)).

## Method

With `g = Sigma~^{1/2} ehat` and `Vs = Sigma~^{1/2} V`, the effective conductivity is the minimum
energy of the cell problem,

    sigma^H_aa / sigma_max = min_w |g + Vs w|^2 = g^T (I - P) g,     P = projector onto range(Vs).

The singular values of `Vs` lie in `[rho^{-1/2}, 1]`, so an even Chebyshev polynomial `f_t` of degree
`d_f = 2t <= sqrt(rho) ln(2/eps_p) + 2` approximates `I - P`, and QSVT applies it with subnormalization 1.
The readout circuit has the all-zero amplitude `A = g^T f_t(Vs Vs^T) g >= 0`, and
`(1 + e_t) A - e_t sbar` is within `e_t sbar` of `sigma^H / sigma_max`. Amplitude estimation reads it out
with `O(sqrt(rho) log(1/eps) / (eps v_f))` applications of `U_V`, independent of the mesh.

## Install

    python -m pip install -r requirements.txt

## Use

    python example_matrix_free.py     # square at N = 1024: cell problem, least squares, polynomial (16 s)
    python example_qiskit.py          # the whole readout circuit at N = 4, state vector (2 s)
    python makeTables.py              # every table of the paper, numbers asserted (about 40 s)
    python makeFigures.py             # figs/cells.pdf, figs/readout_poly.pdf

## Layout

| file | contents |
|---|---|
| `quantumpoissonreadout/cells.py` | the three test cells (square, cross, laminate), matrix-free operators, material oracles, the finite element cell solution |
| `quantumpoissonreadout/polynomial.py` | the range-projection polynomial `f_t`, its QSVT phases, and the inverse polynomial `q_t` of the baseline |
| `quantumpoissonreadout/circuits.py` | the readout as one Qiskit circuit (square), with `Q(k)` from uniformly controlled rotations |
| `quantumpoissonreadout/emulation.py` | the readout amplitude with the gate-level arithmetic `U_V`, and amplitude-estimation sampling |
| `quantumpoissonreadout/readout.py` | the three levels of the readout and the cost against the readout through the inverse |
| `makeTables.py`, `makeFigures.py` | the tables and figures of the paper |
