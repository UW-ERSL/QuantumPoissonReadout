"""emulation -- the readout amplitude with the gate-level U_V of the inversion paper.

The fixed-point arithmetic of U_V needs a work register of 2 F^2 + 35 F + 47 qubits, too many for a
state vector of the whole readout. The gate-level circuit is therefore simulated once per wavenumber
(IsometryCircuit.simulate), which gives the 16 x 16 unitary Q~(k) that it applies, and the readout is
applied to a state vector of shape (position N^2, local 16, s_a, s_b, s_c, z) with Q~(k) inserted
between FFTs. exact_unitaries gives unitaries Q(k) with exact first columns and
||Q(k) - Q~(k)||_2 = delta(k), for the comparison with an exact U_V.
"""
import numpy as np

from quantuminversionpoisson.isometry_circuit import T_matrix, vhat_exact, Q_exact

from .cells import chi

Q_ZERO = np.kron(np.array([[0, 1], [1, 0]]), np.eye(8))        # k = 0: |0> -> |8>


def exact_Q(N):
    """Exact-angle Q(k) for every k."""
    Q = np.zeros((N * N, 16, 16), complex)
    Q[0] = Q_ZERO
    for kk in range(1, N * N):
        Q[kk] = Q_exact(N, *divmod(kk, N))
    return Q


def _plane_rotation_minus_identity(x, y):
    """R - I for the real rotation R with R x = y in span{x, y}, identity elsewhere."""
    w = y - (y @ x) * x
    s = np.linalg.norm(w)
    if s < 1e-300:
        return np.zeros((len(x), len(x)))
    B = np.stack([x, w / s], 1)
    cm1 = -0.5 * np.sum((x - y) ** 2)
    return B @ np.array([[cm1, -s], [s, cm1]]) @ B.T


def exact_unitaries(N, Q):
    """Unitaries with the exact first column Vhat(k), within delta(k) of the gate-level Q~(k)."""
    T, Qe = T_matrix(), Q.copy()
    for kk in range(1, N * N):
        k1, k2 = divmod(kk, N)
        v = np.concatenate([vhat_exact(N, k1, k2), np.zeros(8)])
        P = np.exp(1j * np.pi * (k1 + k2) / N) * T
        x, y = (P.conj().T @ Q[kk][:, 0]).real, (P.conj().T @ v).real
        Qe[kk] = Q[kk] + P @ _plane_rotation_minus_identity(x, y) @ P.conj().T @ Q[kk]
    return Qe


class Emulator:
    """Readout of the square with a given set of Q(k); state axes (N^2, 16, s_a, s_b, s_c, z)."""
    PI_G = (slice(None), slice(0, 8), slice(None), 0, slice(None))     # gradient subspace
    PI_N = (slice(None), 0, slice(None), 0, slice(None))               # nodal subspace

    def __init__(self, N, rho, Q):
        self.N, self.Q = N, Q
        c = chi("square", N)
        self.D = 2 * c - 1                                              # -R_chi
        g0, g1 = (1 + rho ** -0.5) / 2, (1 - rho ** -0.5) / 2         # Sigmatilde^{1/2}
        self.R = np.array([[np.sqrt(g0), -np.sqrt(g1)], [np.sqrt(g1), np.sqrt(g0)]])
        self.sbar = c.mean() + (1 - c.mean()) / rho
        self.prep = np.zeros((N * N, 16)); self.prep[:, 0:8:2] = 1 / (2 * N)   # ehat_x

    def uv(self, s, dag=False):
        N = self.N
        x = np.fft.fft2(s.reshape(N, N, 16, -1), axes=(0, 1), norm="ortho").reshape(N * N, 16, -1)
        Qk = self.Q.conj().transpose(0, 2, 1) if dag else self.Q
        x = np.einsum("kij,kjr->kir", Qk, x).reshape(N, N, 16, -1)
        return np.fft.ifft2(x, axes=(0, 1), norm="ortho").reshape(s.shape)

    def usig(self, s, anc):
        x = np.moveaxis(s, anc, 0)
        y = np.einsum("ij,j...->i...", self.R, x)
        y[1] = y[1] * self.D.reshape(-1, *([1] * (y[1].ndim - 1)))
        y = np.einsum("ji,j...->i...", self.R, y)
        return np.moveaxis(y, 0, anc)

    @staticmethod
    def refl(s, phi, region):
        """e^{i sgn phi (2 Pi - I)}, sgn = +1 on z = 0 and -1 on z = 1."""
        out = np.empty_like(s)
        for zi, sg in ((0, 1.0), (1, -1.0)):
            x = s[..., zi]
            y = x * np.exp(-1j * sg * phi)
            y[region] = x[region] * np.exp(1j * sg * phi)
            out[..., zi] = y
        return out

    def amplitude(self, phi):
        """All-zero amplitude of the readout circuit."""
        s = np.zeros((self.N ** 2, 16, 2, 2, 2, 2), complex)
        s[:, :, 0, 0, 0, 0] = self.prep
        s = self.usig(s, 2)
        s[..., 1] = s[..., 0] / np.sqrt(2); s[..., 0] = s[..., 0] / np.sqrt(2)
        d = len(phi) - 1
        for j in range(d, -1, -1):
            first = (d - j) % 2 == 0
            s = self.refl(s, phi[j], self.PI_G if first else self.PI_N)
            if j > 0:
                s = self.uv(self.usig(s, 3), dag=True) if first else self.usig(self.uv(s), 3)
        s = (s[..., 0] + s[..., 1]) / np.sqrt(2)
        s = self.usig(s, 4)
        return np.vdot(self.prep, s[:, :, 0, 0, 0])


def ae_samples(A, M, runs, rng):
    """Canonical amplitude estimation of the amplitude A >= 0 with M evaluation points, sampled
    from its exact output distribution (Brassard et al., Thm. 11): estimates of A."""
    om = np.arcsin(A) / np.pi
    y = np.arange(M)

    def qpe(w):
        dl = w - y / M
        s = np.sin(np.pi * dl)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(s) < 1e-15, 1.0, np.sin(M * np.pi * dl) ** 2 / (M ** 2 * s ** 2))
    P = 0.5 * (qpe(om) + qpe(1 - om))
    P /= P.sum()
    return np.sin(np.pi * rng.choice(M, size=runs, p=P) / M)
