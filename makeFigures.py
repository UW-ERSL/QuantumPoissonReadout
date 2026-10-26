"""makeFigures.py -- the figures of the paper, written to figs/ as PDF.

    python makeFigures.py            both figures (a few seconds)
    python makeFigures.py poly       just that one

figs/cells.pdf          (fig:cells, Sec. 5.1)   the three test cells at N = 16
figs/readout_poly.pdf   (fig:poly, Sec. 3.2)    the range-projection polynomial f_t, rho = 10
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantumpoissonreadout import GEOMETRIES, chi, poly_plan, f_poly

plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "pdf.fonttype": 42})
EPS_P = 1e-2 / 3


def draw_cells():
    fig, ax = plt.subplots(1, 3, figsize=(5.0, 1.9))
    Nc = 16
    for a, g in zip(ax, GEOMETRIES):
        img = chi(g, Nc).reshape(Nc, Nc).T                          # rows y, columns x
        a.imshow(img, origin="lower", cmap="Greys", vmin=-0.4, vmax=1.3,
                 extent=(0, 1, 0, 1), interpolation="nearest")
        for k in range(Nc + 1):
            a.axhline(k / Nc, color="0.85", lw=0.3); a.axvline(k / Nc, color="0.85", lw=0.3)
        a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.tick_params(labelsize=7)
        a.set_title(f"{g}, $v_f = {chi(g, Nc).mean():.4g}$".replace("0.2344", "15/64")
                    .replace("0.25", "1/4").replace("0.5", "1/2"), fontsize=8)
    fig.tight_layout(); fig.savefig("figs/cells.pdf"); print("    figs/cells.pdf")


def draw_poly(rho=10.0):
    t, d, e = poly_plan(rho, EPS_P)
    x = np.linspace(0, 1, 2001)
    f = f_poly(rho, t)(x)
    assert f.min() >= -1e-15 and abs(f[0] - 1) < 1e-12
    assert f[x >= rho ** -0.5].max() <= 2 * e / (1 + e) + 1e-15        # Prop. poly
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for a in ax:
        a.plot(x, f, color="#1f4e79", lw=1.4)
        a.axvspan(0, rho ** -0.5, color="0.9", lw=0)
        a.axvline(rho ** -0.5, color="0.4", lw=0.8, ls="--")
        a.set_xlabel(r"singular value $x$ of $\mathbf{V}_{\Sigma}$")
        a.set_xlim(0, 1)
    ax[0].set_ylim(-0.05, 1.05); ax[0].set_ylabel(r"$f_t(x)$")
    ax[0].text(0.5 * rho ** -0.5, 0.12, "gap", ha="center", fontsize=9)
    ax[1].set_ylim(-0.1 * e, 2.6 * e)
    ax[1].axhline(2 * e / (1 + e), color="0.4", lw=0.8, ls=":")
    ax[1].set_ylabel(r"$f_t(x)$, enlarged")
    fig.tight_layout(); fig.savefig("figs/readout_poly.pdf")
    print(f"    figs/readout_poly.pdf   (t = {t}, d_f = {d}, e_t = {e:.3e})")


FIGURES = {"cells": draw_cells, "poly": draw_poly}

if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    for name in [a for a in sys.argv[1:] if not a.startswith("-")] or list(FIGURES):
        FIGURES[name]()
