"""quantumpoissonreadout -- least-squares readout of the effective conductivity of periodic
two-phase conduction cells, on the block encodings of quantuminversionpoisson.

    from quantumpoissonreadout import Cell, sigma_fe, ls_exact, ls_poly, qiskit_readout
    cell = Cell("square", N=256, rho=10.0)
    print(sigma_fe(cell), ls_exact(cell), ls_poly(cell, eps_p=1e-2 / 3))
    print(qiskit_readout(N=4))                  # the whole readout circuit, state vector
"""
from .cells import Cell, GEOMETRIES, chi, sigma_fe, check_oracle, uv_counts
from .polynomial import poly_plan, f_poly, even_phases, q_poly
from .circuits import readout_circuit, counts
from .readout import (ls_exact, ls_poly, qiskit_readout, arithmetic_readout, bits_for,
                      amplitude_estimation, ls_cost, baseline_cost)

__all__ = ["Cell", "GEOMETRIES", "chi", "sigma_fe", "check_oracle", "uv_counts", "poly_plan",
           "f_poly", "even_phases", "q_poly", "readout_circuit", "counts", "ls_exact", "ls_poly",
           "qiskit_readout", "arithmetic_readout", "bits_for", "amplitude_estimation", "ls_cost",
           "baseline_cost"]
