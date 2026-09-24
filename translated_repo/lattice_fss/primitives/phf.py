"""
primitives/phf.py
=================
Programmable Hash Function and related primitives.

  EncodingFunction         : FRD encoding  (Definition 2)
  PHFKey                   : key container (standard / trapdoor)
  ProgrammableHashFunction : Zhang, Chen, Zhang (CRYPTO 2016)
  AbortResistantHashFamily : Waters (EUROCRYPT 2005)
"""
from __future__ import annotations
import math, hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ..core.utils import mat_mul_mod, mat_vec_mod, LatticeUtils

class EncodingFunction:
    """FRD encoding function. Paper §II.D, Definition 2."""

    def __init__(self, n: int, q: int):
        self.n = n
        self.q = q

    def evaluate(self, u) -> np.ndarray:
        if np.isscalar(u):
            diag_vals = [(int(u) + i) % self.q for i in range(self.n)]
        else:
            u = np.asarray(u, dtype=np.int64)
            diag_vals = [int(u[i % len(u)]) % self.q for i in range(self.n)]
        return np.diag(diag_vals).astype(np.int64)

    def __call__(self, u) -> np.ndarray:
        return self.evaluate(u)


@dataclass
class PHFKey:
    """PHF public key. K is a list of public matrices; td holds trapdoor information."""
    K: np.ndarray                        # stack of public matrices K_i (ell_PHF × n × nk)
    td: Optional[np.ndarray] = None      # trapdoor scalars h_i ∈ Z_q^{ell_PHF}
    A_ref: Optional[np.ndarray] = None   # reference matrix A from TrapGen
    G_ref: Optional[np.ndarray] = None   # gadget matrix G


class ProgrammableHashFunction:
    """
    Lattice-based PHF — construction from Zhang, Chen, Zhang (CRYPTO 2016) §3.

    ─── Construction overview (Zhang et al. 2016, §3.1) ────────────────
    Parameters: n, m̄, nk, q, β, ℓ_PHF (number of hash matrices)

    H.Gen(1^κ):
      For each i ∈ [ℓ_PHF], independently sample K_i ← U(Z_q^{n×nk}).
      Key K = (K_0, ..., K_{ℓ_PHF-1}).

    H.Eval(K, M):
      Convert message M to bytes and map to a bit string b ∈ {0,1}^ℓ_PHF via SHA-256.
      HK(M) = Σ_{i: b_i=1} K_i  +  Σ_{i: b_i=0} (-K_i)  mod q
            = Σ_i c_i K_i,  c_i ∈ {+1, -1}
      ─ This ensures HK(M) - HK(M') is statistically close to uniform for all M, M'.

    H.TrapGen(1^κ, A, G):
      Sample h = (h_0,...,h_{ℓ_PHF-1}) ← U(Z_q^{ℓ_PHF}) (trapdoor).
      For each i, sample R_i ← D_{Z^{m̄×nk}, β̄} and set
        K_i = A R_i + h_i G  ∈ Z_q^{n×nk}.
      K' = (K_0,...), td = (h, R_0,...,R_{ℓ_PHF-1}).
      By definition, HK'(M) = A (Σ c_i R_i) + (Σ c_i h_i) G = A R_M + S_M G
      (Correctness: R_M = Σ c_i R_i, S_M = Σ c_i h_i ∈ Z_q scaled by I_n)

    H.TrapEval(td, K', M):
      Compute c_i from M.
      R_M = Σ c_i R_i  ∈ Z^{m̄×nk}
      S_M = (Σ c_i h_i mod q) · I_n  ∈ Z_q^{n×n}
      Return: (R_M, S_M)  → HK'(M) = A R_M + S_M G  (Correctness holds)
    ─────────────────────────────────────────────────────────────────

    Security: the PHF (u,v,β,γ,δ) conditions hold under the hardness of ISIS.
    (Zhang et al. 2016 Theorem 3.2).
    """

    def __init__(self, n: int, m_bar: int, nk: int, q: int, beta: float,
                 ell_phf: int = 128):
        """
        Parameters
        ----------
        n       : lattice dimension
        m_bar   : number of rows in PHF (typically m_bar = m = 2n+nk)
        nk      : number of columns of hash value
        q       : modulus (prime)
        beta    : Gaussian parameter β̄ (for sampling R_i)
        ell_phf : number of hash matrices (bit width of the message space)
        """
        self.n       = n
        self.m_bar   = m_bar
        self.nk      = nk
        self.q       = q
        self.beta    = beta
        self.ell_phf = ell_phf   # ℓ_PHF

    # ── Message → ±1 coefficient vector conversion ────────────────────

    def _msg_to_bits(self, M) -> np.ndarray:
        """
        Encodes M as bytes and maps to {+1,-1}^{ℓ_PHF} via SHA-256.
        b_i = +1 if SHA-256(M)[i] = 1, -1 otherwise.
        """
        raw = str(M).encode() if not isinstance(M, (bytes, bytearray)) else M
        # Obtain at least ell_phf bits via repeated hashing
        bits = []
        seed = raw
        while len(bits) < self.ell_phf:
            digest = hashlib.sha256(seed).digest()
            for byte in digest:
                for bit_pos in range(8):
                    bits.append(1 if (byte >> bit_pos) & 1 else -1)
            seed = hashlib.sha256(seed).digest()
        return np.array(bits[: self.ell_phf], dtype=np.int64)

    # ── H.Gen ────────────────────────────────────────────────────────

    def gen(self) -> PHFKey:
        """
        H.Gen(1^κ):
          K_i ← U(Z_q^{n×nk})  for i = 0,...,ℓ_PHF-1
        """
        K_stack = np.array([
            LatticeUtils.sample_uniform_matrix(self.n, self.nk, self.q)
            for _ in range(self.ell_phf)
        ], dtype=np.int64)   # shape: (ell_phf, n, nk)
        return PHFKey(K=K_stack)

    # ── H.Eval ───────────────────────────────────────────────────────

    def eval(self, key: PHFKey, M) -> np.ndarray:
        """
        H.Eval(K, M) = Σ_i c_i K_i  mod q,   c_i ∈ {+1, -1}

        Returns
        -------
        HK(M) ∈ Z_q^{n×nk}
        """
        c = self._msg_to_bits(M)          # (ell_phf,) ∈ {+1,-1}
        K_stack = np.asarray(key.K, dtype=np.int64)  # (ell_phf, n, nk)
        result = np.zeros((self.n, self.nk), dtype=np.int64)
        for i in range(self.ell_phf):
            result = np.mod(result + int(c[i]) * K_stack[i], self.q)
        return result

    # ── H.TrapGen ────────────────────────────────────────────────────

    def trap_gen(
        self, A: np.ndarray, G: np.ndarray
    ) -> Tuple['PHFKey', dict]:
        """
        H.TrapGen(1^κ, A, G):
          h_i ← U(Z_q)  for i = 0,...,ℓ_PHF-1      (trapdoor scalars)
          R_i ← D_{Z^{m̄×nk}, β̄}                    (short matrix)
          K_i = A R_i + h_i G  ∈ Z_q^{n×nk}

        Correctness:
          HK'(M) = A (Σ c_i R_i) + (Σ c_i h_i) G = A R_M + S_M G  ✓

        Returns
        -------
        key : PHFKey (K_stack, td=h vector, A_ref=A, G_ref=G)
        td  : dict {'h': h, 'Rs': R_stack}
        """
        A = np.asarray(A, dtype=np.int64)
        G = np.asarray(G, dtype=np.int64)

        h = np.array([secrets.randbelow(self.q) for _ in range(self.ell_phf)],
                     dtype=np.int64)
        R_stack = np.array([
            LatticeUtils.sample_discrete_gaussian_matrix(self.m_bar, self.nk, self.beta)
            for _ in range(self.ell_phf)
        ], dtype=np.int64)   # (ell_phf, m_bar, nk)

        # K_i = A R_i + h_i G  mod q
        K_stack = np.empty((self.ell_phf, self.n, self.nk), dtype=np.int64)
        for i in range(self.ell_phf):
            AR_i  = mat_mul_mod(A, R_stack[i], self.q)
            hiG   = np.mod(int(h[i]) * G, self.q)
            K_stack[i] = np.mod(AR_i + hiG, self.q)

        td = {'h': h, 'Rs': R_stack}
        key = PHFKey(K=K_stack, td=h, A_ref=A, G_ref=G)
        return key, td

    # ── H.TrapEval ───────────────────────────────────────────────────

    def trap_eval(
        self, td: dict, key: 'PHFKey', M
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        H.TrapEval(td, K', M) → (R_M, S_M)

        R_M = Σ_i c_i R_i  ∈ Z^{m̄×nk}
        S_M = (Σ_i c_i h_i mod q) · I_n  ∈ Z_q^{n×n}

        Correctness (Definition 3 of the paper):
          HK'(M) = A R_M + S_M G  ✓

        Condition for S_M ∈ GL_n(Z_q) ∪ {0}:
          Σ c_i h_i ≡ 0 (mod q)  → S_M = 0
          Σ c_i h_i ≢ 0 (mod q)  → S_M = s · I_n (s invertible scalar) ∈ GL_n(Z_q)
        """
        c = self._msg_to_bits(M)

        if isinstance(td, dict):
            h      = np.asarray(td['h'],  dtype=np.int64)
            R_stack = np.asarray(td['Rs'], dtype=np.int64)
        else:
            # Fallback (legacy td format)
            h = np.zeros(self.ell_phf, dtype=np.int64)
            R_stack = np.zeros((self.ell_phf, self.m_bar, self.nk), dtype=np.int64)

        # R_M = Σ c_i R_i
        R_M = np.zeros((self.m_bar, self.nk), dtype=np.int64)
        for i in range(self.ell_phf):
            R_M = np.mod(R_M + int(c[i]) * R_stack[i], self.q)

        # S_M = (Σ c_i h_i mod q) · I_n
        s_scalar = int(np.dot(c.astype(object), h.astype(object))) % self.q
        S_M = np.mod(s_scalar * np.eye(self.n, dtype=np.int64), self.q)

        return R_M, S_M

    # ── Self-verification of Correctness ─────────────────────────────────

    def verify_correctness(
        self, key: 'PHFKey', td: dict, M, q: int
    ) -> bool:
        """
        Verifies HK'(M) == A R_M + S_M G  (for debugging).
        """
        HKM   = self.eval(key, M)
        R_M, S_M = self.trap_eval(td, key, M)
        A = np.asarray(key.A_ref, dtype=np.int64)
        G = np.asarray(key.G_ref, dtype=np.int64)
        lhs = np.mod(mat_mul_mod(A, R_M, q) + mat_mul_mod(S_M, G, q), q)
        return bool(np.all(HKM == lhs))


class AbortResistantHashFamily:
    """Waters-type abort-resistant hash family. Paper §II.F, Definition 4."""

    def __init__(self, ell: int, q: int):
        self.ell = ell
        self.q = q

    def gen(self) -> np.ndarray:
        return np.array([secrets.randbelow(self.q) for _ in range(self.ell + 1)], dtype=np.int64)

    def evaluate(self, h: np.ndarray, ID_t: np.ndarray) -> int:
        return int((1 + int(np.dot(h, ID_t))) % self.q)

    def non_abort_probability_bounds(self, Q: int) -> Tuple[float, float]:
        return (1.0 / self.q) * (1.0 - Q / self.q), 1.0 / self.q
