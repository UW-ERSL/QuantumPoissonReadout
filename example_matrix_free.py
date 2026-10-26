from quantumpoissonreadout import Cell, sigma_fe, ls_exact, ls_poly

cell = Cell("square", N=1024, rho=10.0)              # 1,048,576 unknowns
fe = sigma_fe(cell)                                  # cell problem, PCG
ls = ls_exact(cell)                                  # least-squares form, LSQR
est, d, e_t = ls_poly(cell, eps_p=1e-2 / 3)          # polynomial of degree d
print(f"sigma_H (FE)     = {fe:.8f}")
print(f"least squares    = {ls:.8f}")
print(f"degree d         = {d}")
print(f"estimate         = {est:.8f}")
print(f"error / sbar     = {(est - fe) / cell.sbar:.2e}")
