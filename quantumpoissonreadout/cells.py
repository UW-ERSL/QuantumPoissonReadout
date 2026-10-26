"""cells -- the three periodic two-phase test cells, applied matrix-free.

    square     centered square inclusion of side N/2, v_f = 1/4 (isolated conductor)
    cross      two orthogonal bands of width N/8 through the center, v_f = 15/64 (connected)
    laminate   chi = 1 for x < 1/2, v_f = 1/2 (layers normal to x)

Every operator is applied without dense matrices (FFT for functions of K, sparse A = W^{1/2} G),
so the cells reach N = 1024. Each cell has a material oracle controlled(-R_chi) of O(1) gates,
checked on every element. sigma_max = 1 throughout.
"""
import numpy as np
import scipy.sparse as sp

from quantuminversionpoisson.fourier_symbol import symbol_function, apply_mode_function
from quantuminversionpoisson.isometry_circuit import _B, _CORNERS, IsometryCircuit

GEOMETRIES = ("square", "cross", "laminate")


# ---- geometry and material oracle ----------------------------------------------------
def chi(name, N):
    """Phase indicator per element e = i N + j (i along x): 1 in the inclusion."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if name == "square":
        c = (i >= N // 4) & (i < 3 * N // 4) & (j >= N // 4) & (j < 3 * N // 4)
    elif name == "cross":
        band = lambda k: (k >= 7 * N // 16) & (k < 9 * N // 16)
        c = band(i) | band(j)
    elif name == "laminate":
        c = i < N // 2
    else:
        raise ValueError(name)
    return c.ravel().astype(float)


def oracle_network(name):
    """Gate list of controlled(-R_chi) on (s, leading bits of i and j, work bits)."""
    if name == "laminate":
        return [("cz", "s", "i0")]
    g = [("z", "s")]
    if name == "square":                        # chi = (i0 xor i1) and (j0 xor j1)
        comp = [("cx", "i0", "i1"), ("cx", "j0", "j1"), ("ccx", "i1", "j1", "t", 1, 1)]
        return g + comp + [("cz", "s", "t")] + comp[::-1]
    if name == "cross":                         # band: top four bits in {0111, 1000}
        comp = []
        for c in "ij":
            comp += [("cx", f"{c}2", f"{c}3"), ("cx", f"{c}1", f"{c}2"), ("cx", f"{c}0", f"{c}1")]
            comp += [("ccx", f"{c}1", f"{c}2", f"u{c}", 1, 0),
                     ("ccx", f"u{c}", f"{c}3", f"a{c}", 1, 0)]
        mid = [("cz", "s", "ai"), ("cz", "s", "aj"), ("ccz", "s", "ai", "aj")]
        return g + comp + mid + comp[::-1]
    raise ValueError(name)


def oracle_phase(name, n):
    """Simulate oracle_network on every (s, position): phase table and gate counts."""
    N = 2 ** n
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i, j = i.ravel(), j.ravel()
    out = {}
    for s in (0, 1):
        bits = {"s": np.full(N * N, s)}
        for k in range(4):
            bits[f"i{k}"] = (i >> (n - 1 - k)) & 1 if k < n else np.zeros_like(i)
            bits[f"j{k}"] = (j >> (n - 1 - k)) & 1 if k < n else np.zeros_like(j)
        phase = np.ones(N * N)
        for gate in oracle_network(name):
            for w in gate[1:4]:
                if isinstance(w, str):
                    bits.setdefault(w, np.zeros(N * N, int))
            op = gate[0]
            if op == "z":
                phase *= (-1.0) ** bits[gate[1]]
            elif op == "cz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]])
            elif op == "ccz":
                phase *= (-1.0) ** (bits[gate[1]] & bits[gate[2]] & bits[gate[3]])
            elif op == "cx":
                bits[gate[2]] = bits[gate[2]] ^ bits[gate[1]]
            elif op == "ccx":
                a, b, t, pa, pb = gate[1:]
                bits[t] = bits[t] ^ ((bits[a] == pa) & (bits[b] == pb)).astype(int)
        work = [w for w in bits if w[0] in "uat" and w != "s"]
        assert all(not bits[w].any() for w in work), name
        out[s] = phase
    net = oracle_network(name)
    counts = dict(toffoli=sum(g[0] in ("ccx", "ccz") for g in net),
                  cnot=sum(g[0] in ("cx", "cz") for g in net),
                  single=sum(g[0] in ("x", "z") for g in net))
    return out, counts


_ORC = {}


def check_oracle(name):
    """controlled(-R_chi) is I for s = 0 and 2 chi - 1 for s = 1, at N = 16 ... 256; gate counts."""
    if name not in _ORC:
        for n in range(4, 9):
            ph, counts = oracle_phase(name, n)
            assert np.all(ph[0] == 1) and np.array_equal(ph[1], 2 * chi(name, 2 ** n) - 1), (name, n)
        _ORC[name] = counts
    return _ORC[name]


# ---- operators ----------------------------------------------------------------------------
def k_function(f, N):
    """x -> f(K) x by the FFT, zero on the constant field."""
    T = symbol_function(f, N)
    return lambda x: apply_mode_function(T, x)


_A = {}


def A_matrix(N):
    """A = W^{1/2} G (8 N^2 x N^2), row 8 e + 2 g + c for element e, Gauss point g, component c."""
    if N not in _A:
        h = 1 / N
        i, j = np.divmod(np.arange(N * N), N)
        rows, cols, vals = [], [], []
        for g in range(4):
            Bg = _B(g, h)
            for a, (cx, cy) in enumerate(_CORNERS):
                node = ((i + cx) % N) * N + (j + cy) % N
                for c in range(2):
                    rows.append(8 * np.arange(N * N) + 2 * g + c)
                    cols.append(node)
                    vals.append(np.full(N * N, Bg[c, a]))
        _A[N] = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(8 * N * N, N * N)) * (h / 2)
    return _A[N]


class Cell:
    """Two-phase cell: geometry, N, rho; K^chi and the macroscopic gradient, matrix-free."""

    def __init__(self, name, N, rho):
        self.name, self.N, self.rho = name, N, rho
        self.A = A_matrix(N)
        self.chi = chi(name, N)
        self.vf = self.chi.mean()
        self.St = np.repeat(np.where(self.chi == 1, 1.0, 1 / rho), 8)      # Sigmatilde
        self.sbar = self.vf + (1 - self.vf) / rho                          # Voigt bound
        self.reuss = 1 / (self.vf + (1 - self.vf) * rho)                   # Reuss bound
        self.Kih = k_function(lambda e: e ** -0.5, N)
        self.Kinv = k_function(lambda e: 1 / e, N)
        self.n = N * N

    def Kc(self, u):
        """K^chi / sigma_max."""
        return self.A.T @ (self.St * (self.A @ u))

    def ehat(self, a=1):
        """Normalized macroscopic gradient W^{1/2} ebar_a: sqrt(w) = 1/(2N), unit norm."""
        v = np.zeros(8 * self.n)
        v[(a - 1)::2] = 1 / (2 * self.N)
        return v


def pcg(Aop, Pop, b, tol, maxiter=5000):
    """Preconditioned CG on the complement of the constant fields."""
    b = b - b.mean()
    x = np.zeros_like(b)
    r = b.copy()
    z = Pop(r); z -= z.mean()
    p = z.copy()
    rz = r @ z
    nb = np.sqrt(b @ Pop(b))
    for _ in range(maxiter):
        Ap = Aop(p)
        al = rz / (p @ Ap)
        x += al * p
        r -= al * Ap
        z = Pop(r); z -= z.mean()
        rz_new = r @ z
        if np.sqrt(abs(rz_new)) <= tol * nb:
            return x
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError("pcg did not converge")


def sigma_fe(cell, a=1):
    """sigma^H_aa / sigma_max from the finite element cell problem (PCG preconditioned by K^+)."""
    e = cell.ehat(a)
    u = pcg(cell.Kc, cell.Kinv, -(cell.A.T @ (cell.St * e)), 1e-13)
    r = e + cell.A @ u
    return float(r @ (cell.St * r))


# ---- resources of the isometry -------------------------------------------------------------
_UV = {}


def uv_counts(N, F):
    """Toffoli gates and qubits of one gate-level U_V with F fractional bits (cached)."""
    if (N, F) not in _UV:
        r = IsometryCircuit(n=int(np.log2(N)), F=F).resources()
        _UV[N, F] = dict(toffoli=r["toffoli_total"], qubits=r["wires"])
    return _UV[N, F]
