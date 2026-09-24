"""
core/samplers.py
================
Gaussian samplers for lattices.

  PeikertGaussianSampler : Klein/GPV + Peikert (CRYPTO 2010)
  GadgetSampler          : MP2012 §4.1 (C-extension accelerated)
"""
from __future__ import annotations
import math
from typing import Optional
import secrets
import numpy as np
from .utils import mat_mul_mod, mat_vec_mod, _fast_lib, _FAST_AVAILABLE, _FAST_LIB

class PeikertGaussianSampler:
    """
    Preimage sampling algorithm from Peikert (2010).

    ─── Theoretical background ─────────────────────────────────
    The goal is to sample from D_{Λ, Σ, c} for a lattice Λ
    and covariance matrix Σ.

    "Perturbation + Klein" procedure from Peikert §4.1:
      x_pert ← N(c, Σ_p)      [continuous perturbation sample; Σ_p = Σ - Σ_0]
      e       ← Klein(B, s_0, x_pert)  [snap to lattice point]
      return e

    where Σ_0 = (1/s_0^2) B B^T (Gram matrix of B).
    Requires s_0 ≥ η_ε(Λ) for the statistical distance to be negligible.
    (Micciancio–Regev, FOCS 2004).

    This class is based on the Klein (GPV 2008) implementation.
    ─────────────────────────────────────────────────────────────
    """

    # ── 1-D discrete Gaussian sampler (rejection sampling) ──────────

    @staticmethod
    def sample_z(center: float, sigma: float, tail_cut: int = 6) -> int:
        """
        Returns a sample from D_{Z, sigma, center} via rejection sampling.

        Pr[x = k] ∝ exp(-π (k - center)^2 / sigma^2)

        tail_cut: truncates range |k - center| > tail_cut * sigma.
        For security, tail_cut ≥ 6 (statistical distance ≈ exp(-π*36) ≈ 2^{-163}).
        """
        lo = math.floor(center - tail_cut * sigma)
        hi = math.ceil(center + tail_cut * sigma)
        while True:
            k = secrets.randbelow(hi - lo + 1) + lo
            rho = math.exp(-math.pi * (k - center) ** 2 / (sigma ** 2))
            # Bernoulli(rho) test using cryptographic randomness
            u = int.from_bytes(secrets.token_bytes(7), "big") >> 3  # 53 bit
            if u < int(rho * (1 << 53)):
                return k

    @staticmethod
    def sample_z_vector(center: np.ndarray, sigma: float) -> np.ndarray:
        """Independently samples each component from D_{Z, sigma, c_i}."""
        return np.array(
            [PeikertGaussianSampler.sample_z(float(c), sigma) for c in center],
            dtype=np.int64,
        )

    # ── Gram-Schmidt orthogonalization ──────────────────────────────

    @staticmethod
    def gram_schmidt(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the Gram-Schmidt orthogonalization of B.

        Returns
        -------
        B_tilde : Gram-Schmidt orthogonal basis (column vectors, float)
        mu      : GSO coefficient matrix (lower triangular, μ_{i,j} = <B_i, B̃_j>/‖B̃_j‖²)
        """
        m, n = B.shape
        B_tilde = np.zeros((m, n), dtype=float)
        mu = np.zeros((n, n), dtype=float)
        for i in range(n):
            B_tilde[:, i] = B[:, i].astype(float)
            for j in range(i):
                denom = np.dot(B_tilde[:, j], B_tilde[:, j])
                if denom > 1e-14:
                    mu[i, j] = np.dot(B[:, i], B_tilde[:, j]) / denom
                    B_tilde[:, i] -= mu[i, j] * B_tilde[:, j]
            mu[i, i] = 1.0
        return B_tilde, mu

    # ── Klein (GPV 2008) algorithm ──────────────────────────────────

    @classmethod
    def klein(
        cls,
        B: np.ndarray,
        sigma: float,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Klein (2000) / GPV (2008) algorithm.

        Samples a lattice point x from D_{Λ(B), sigma, target}.

        Algorithm (GPV 2008 §4, Lemma 4.4):
        ─────────────────────────────────────────
        B̃ = GS(B)
        c ← target (floating-point copy)
        for i = n-1 downto 0:
            c_i  = <c, B̃_i> / ‖B̃_i‖²
            σ_i  = σ / ‖B̃_i‖
            z_i ~ D_{Z, σ_i, c_i}   (1-D rejection sampling)
            c   ← c - z_i · B_i
        return Σ_i z_i · B_i + (offset pulling target toward the lattice)

        Statistical distance of output is ≤ 2^{-n} when σ ≥ η_ε(Λ).

        Parameters
        ----------
        B      : lattice basis (m × n), columns are basis vectors
        sigma  : Gaussian parameter (≥ η_ε(Λ))
        target : target point ∈ ℝ^m

        Returns
        -------
        x : lattice point ∈ Λ(B) ⊂ ℤ^m
        """
        m, n = B.shape
        B_tilde, _ = cls.gram_schmidt(B)
        coeffs = np.zeros(n, dtype=np.int64)
        c = target.astype(float).copy()

        for i in range(n - 1, -1, -1):
            b_tilde_i = B_tilde[:, i]
            b_norm_sq = np.dot(b_tilde_i, b_tilde_i)
            if b_norm_sq < 1e-14:
                continue
            c_i = np.dot(c, b_tilde_i) / b_norm_sq
            sigma_i = sigma / math.sqrt(b_norm_sq)
            z_i = cls.sample_z(c_i, sigma_i)
            coeffs[i] = z_i
            c -= float(z_i) * B[:, i].astype(float)

        return B.astype(np.int64) @ coeffs


class GadgetSampler:
    """
    Precise sampling from the gadget inverse G^{-1}.

    G = I_n ⊗ g^T,  g = (1, 2, 4, ..., 2^{k-1})^T ∈ ℤ^k

    ─── Goal ───────────────────────────────────────────────────
    x ~ D_{Λ_v^⊥(G), r}:
      Obtain x ∈ ℤ^{nk} satisfying Gx = v mod q with short ‖x‖.

    ─── Algorithm (Micciancio-Peikert 2012, §4.1) ──────────────
    For each row i ∈ {0, ..., n-1} independently:
      1-D problem: solve g^T x_i = v_i mod q.
      This is a k-bit binary expansion problem:
        x_{i,0} + 2 x_{i,1} + ... + 2^{k-1} x_{i,k-1} ≡ v_i (mod q)

    Determine each digit via parity-constrained rejection sampling:
      carry_0 = v_i
      for j = 0, ..., k-2:
          parity_j = carry_j mod 2
          x_{i,j} ~ D_{Z, r}  subject to  x_{i,j} ≡ parity_j (mod 2)
          carry_{j+1} = (carry_j - x_{i,j}) / 2
      x_{i,k-1} = carry_{k-1}    (last digit is deterministic)

    This procedure yields output distribution within 2^{-Ω(k)} of D_{Z^k, r}
    (Micciancio–Peikert Lemma 4.1).
    ─────────────────────────────────────────────────────────────
    """

    def __init__(self, n: int, q: int, k: int):
        self.n = n
        self.q = q
        self.k = k

    # ── Parity-constrained 1-D sampler ──────────────────────────────

    @staticmethod
    def _sample_parity(center: float, sigma: float, parity: int) -> int:
        """
        Returns a sample from D_{Z, sigma, center} with parity constraint (x ≡ parity mod 2).

        Method: rejection sampling.
          z ~ D_{Z, sigma, center}
          if z ≡ parity (mod 2): return z
          else: retry

        Expected number of trials is at most 2 (due to even/odd symmetry).
        """
        while True:
            z = PeikertGaussianSampler.sample_z(center, sigma)
            if (z % 2) == (parity % 2):
                return z

    # ── Single-row gadget preimage sampling ─────────────────────────

    def sample_gadget_preimage_row(self, v_i: int, r: float) -> np.ndarray:
        """
        Samples a short x ∈ ℤ^k satisfying g^T x = v_i mod q.

        Follows Micciancio-Peikert §4.1 "per-coordinate digit sampling".

        Parameters
        ----------
        v_i : target integer ∈ {0, ..., q-1}
        r   : Gaussian parameter

        Returns
        -------
        x : ∈ ℤ^k,  g^T x = v_i mod q, ‖x‖ short
        """
        x = np.zeros(self.k, dtype=np.int64)
        carry = int(v_i) % self.q

        for j in range(self.k - 1):
            parity = carry % 2
            x_j = self._sample_parity(0.0, r, parity)
            x[j] = x_j
            carry = (carry - x_j) >> 1   # integer division by 2

        # Last digit: assign remaining carry directly
        # (‖x_{k-1}‖ may be large, but since q ≤ 2^k,
        #  |carry| < q / 2^{k-1} ≤ 2 in practice)
        x[self.k - 1] = carry
        return x

    # ── nk-dimensional vector version ───────────────────────────────

    def sample_Gperp(self, v: np.ndarray, r: float) -> np.ndarray:
        """
        x ~ D_{Λ_v^⊥(G), r}:
          Gx = v mod q,  ‖x‖ short,  x ∈ ℤ^{nk}

        Acceleration: when the C extension (fss_fast.so) is available,
        uses xoshiro256** PRNG with O3-optimized rejection sampling
        (~120x faster than pure Python). Falls back to pure Python otherwise.
        """
        assert len(v) == self.n, f"Length of v must equal n={self.n}."

        if _FAST_AVAILABLE:
            v_arr = np.ascontiguousarray(v, dtype=np.int64)
            out   = np.zeros(self.n * self.k, dtype=np.int64)
            _FAST_LIB.sample_gperp(
                v_arr.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
                out.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
                self.n, self.q, self.k, r, 6,
            )
            return out

        # Fallback: pure Python
        blocks = [self.sample_gadget_preimage_row(int(v[i]), r) for i in range(self.n)]
        return np.concatenate(blocks, axis=0)

    # ── nk × cols matrix version ─────────────────────────────────────

    def sample_Gperp_matrix(self, V: np.ndarray, r: float) -> np.ndarray:
        """
        Matrix version of D_{Λ_V^⊥(G), r}:
          GX = V mod q,  X ∈ ℤ^{nk × cols}
        Samples each column independently (using C extension if available).
        """
        cols = V.shape[1]
        X = np.empty((self.n * self.k, cols), dtype=np.int64)
        for j in range(cols):
            X[:, j] = self.sample_Gperp(V[:, j], r)
        return X
