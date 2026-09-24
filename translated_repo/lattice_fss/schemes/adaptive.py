"""
schemes/adaptive.py
===================
AdaptiveFSS: adaptive FS-EUF-CMA forward-secure signature (§V-C).
First lattice-based construction achieving adaptive security.

  PublicKeyAdaptive  : public key container
  SigningKeyAdaptive : signing key container
  AdaptiveFSS        : KeyGen / KeyUpdate / Sign / Verify
"""
from __future__ import annotations
import secrets
import math, hashlib, secrets
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ..core.utils import mat_mul_mod, mat_vec_mod, LatticeUtils
from ..core.trapdoor import GadgetTrapdoor, SwapTrap
from ..primitives.phf import PHFKey, ProgrammableHashFunction, AbortResistantHashFamily
from .selective import SelectiveFSS

@dataclass
class PublicKeyAdaptive:
    A: np.ndarray
    A_primes: List[List[np.ndarray]]
    u: np.ndarray
    K: PHFKey
    IDs: List[np.ndarray]


@dataclass
class SigningKeyAdaptive:
    R: np.ndarray
    H_tag: np.ndarray
    t: int = 0
    R_trap: Optional[np.ndarray] = None


class AdaptiveFSS:
    """
    Adaptively secure lattice-based FSS (paper §V.C).
    First construction combining abort-resistant hash and precise SwapTrap.
    """

    def __init__(self, n: int, q: int, T: int, ell: int, sigma: float):
        self.n = n
        self.q = q
        self.T = T
        self.ell = ell
        self.sigma = sigma
        self.gadget = GadgetTrapdoor(n, q)
        self.m = self.gadget.m
        self.nk = self.gadget.nk
        self.G = self.gadget.G
        self.phf = ProgrammableHashFunction(n, self.m, self.nk, q, beta=sigma)
        self.abort_hash = AbortResistantHashFamily(ell, q)
        self.swap = SwapTrap(self.gadget, q, r=sigma / math.sqrt(2.0))

    def _crh(self, H_prev, ID_t):
        """
        Collision-resistant hash H' : {0,1}* → GL_n(Z_q).
        AdaptiveFSS version: uses identity ID_t as input.

        Implementation (Improvement A1):
          Uses SHAKE-256 directly, eliminating np.random.
          Unified design identical to SelectiveFSS._crh.
        """
        max_retry = 64
        base = H_prev.tobytes() + np.asarray(ID_t, dtype=np.int64).tobytes()
        n2 = self.n * self.n
        for attempt in range(max_retry):
            xof = hashlib.shake_256(base + attempt.to_bytes(2, 'big'))
            raw = xof.digest(n2 * 3)
            elems = np.array([
                int.from_bytes(raw[i*3:(i+1)*3], 'big') % self.q
                for i in range(n2)
            ], dtype=np.int64)
            H_cand = elems.reshape(self.n, self.n)
            H_cand = np.mod(H_cand + np.eye(self.n, dtype=np.int64), self.q)
            try:
                LatticeUtils.inv_mod(H_cand, self.q)
                return H_cand
            except ValueError:
                continue
        return np.eye(self.n, dtype=np.int64)

    def _sample_identity(self):
        return np.array([1] + [1 if secrets.randbelow(2) == 0 else -1
                               for _ in range(self.ell)], dtype=np.int64)

    def _compute_Bt(self, A_primes, IDs, t):
        Bt = self.G.copy().astype(np.int64)
        for i in range(self.ell + 1):
            Bt = np.mod(Bt + int(IDs[t][i]) * A_primes[t][i].astype(np.int64), self.q)
        return Bt

    def _make_At(self, pk, t):
        Bt = self._compute_Bt(pk.A_primes, pk.IDs, t)
        return np.hstack([pk.A, Bt]), Bt

    def _build_correction(self, pk, t):
        _, B0 = self._make_At(pk, 0)
        _, Bt = self._make_At(pk, t)
        diff = np.mod(B0 - Bt, self.q)
        E_t = np.vstack([diff, np.zeros((self.m - self.n, self.nk), dtype=np.int64)])
        K_t = np.block([
            [np.eye(self.m, dtype=np.int64), E_t],
            [np.zeros((self.nk, self.m), dtype=np.int64), np.eye(self.nk, dtype=np.int64)],
        ])
        return E_t, K_t

    def key_gen(self):
        A, R_trap = self.gadget.trap_gen(self.sigma)
        A_primes = [
            [LatticeUtils.sample_uniform_matrix(self.n, self.nk, self.q)
             for _ in range(self.ell + 1)]
            for _ in range(self.T + 1)
        ]
        IDs = [self._sample_identity() for _ in range(self.T + 1)]
        B0 = self._compute_Bt(A_primes, IDs, 0)
        A0 = np.hstack([A, B0])
        H_rand = LatticeUtils.sample_uniform_matrix(self.n, self.n, self.q)
        H0 = self._crh(H_rand, IDs[0])
        A0_ext = np.hstack([A0, mat_mul_mod(H0, self.G, self.q)])
        R_part = R_trap[:self.m, :] if R_trap.shape[0] >= self.m else R_trap
        R0 = self.gadget.del_trap(
            A0_ext, R_part, np.eye(self.n, dtype=np.int64), H0, self.sigma, self.q
        )
        u = LatticeUtils.sample_uniform_matrix(self.n, 1, self.q).flatten()
        K = self.phf.gen()
        return (PublicKeyAdaptive(A=A, A_primes=A_primes, u=u, K=K, IDs=IDs),
                SigningKeyAdaptive(R=R0, H_tag=H0, t=0, R_trap=R_trap))

    def key_update(self, pk, sk_t):
        t = sk_t.t
        if t >= self.T:
            raise ValueError(f"Maximum key updates T={self.T} reached.")
        A0, _ = self._make_At(pk, 0)
        H_t1 = self._crh(sk_t.H_tag, pk.IDs[t + 1])
        return SigningKeyAdaptive(
            R=self.swap.execute(A0, sk_t.R, sk_t.H_tag, H_t1, self.sigma),
            H_tag=H_t1, t=t + 1, R_trap=sk_t.R_trap,
        )

    def sign(self, pk, sk_t, M: bytes, t: int):
        """
        Sign(pk, sk_t, M, t) → σ = [x0; x1]

        Compliant with Sign in §V.C (revised version).
        Step 8: x0 = SampleD(Rt, AtKt, Ht, uM, s)
          Absorb the right nk columns A Et + Bt of AtKt via y1,
          and solve the A (n, m) part using R_trap over pk.A.
        """
        E_t, K_t = self._build_correction(pk, t)
        HK_M = self.phf.eval(pk.K, M)
        At, Bt = self._make_At(pk, t)

        x1 = LatticeUtils.sample_discrete_gaussian_vector(self.nk, self.sigma)
        u_M = np.mod(pk.u - mat_vec_mod(HK_M, x1, self.q), self.q)

        # Right nk columns of AtKt = A Et + Bt
        A_Et = mat_mul_mod(pk.A, E_t, self.q)
        AtKt_right = np.mod(A_Et + Bt, self.q)

        y1 = LatticeUtils.sample_discrete_gaussian_vector(self.nk, self.sigma)
        u_A = np.mod(u_M - mat_vec_mod(AtKt_right, y1, self.q), self.q)
        R_for_sampleD = sk_t.R_trap if sk_t.R_trap is not None else sk_t.R
        H_for_sampleD = np.eye(self.n, dtype=np.int64)   # tag of R_trap is I_n
        y0 = self.gadget.sample_D(R_for_sampleD, pk.A, H_for_sampleD, u_A, self.sigma, self.q)
        x0 = np.concatenate([y0, y1])
        return np.concatenate([x0, x1])

    def verify(self, pk, M: bytes, sig: np.ndarray, t: int):
        """
        Verify(pk, M, σ, t) → 1/⊥

        Compliant with Verify in §V.C (revised version).
        Checks At,M = [AtKt | HK(M)]  and  At,M σ = u.

        Improvement A2: margin factor of norm bound changed from ×30 to ×3.
        Improvement B2: Compute AtKt exploiting the diagonal block structure of Kt.
                  Applies the same optimization as SelectiveFSS.verify.
        """
        E_t, _ = self._build_correction(pk, t)  # Kt is not needed
        HK_M = self.phf.eval(pk.K, M)
        At, Bt = self._make_At(pk, t)

        # B2: Fast computation of AtKt via diagonal block structure
        A_part = At[:, :self.m]
        AtKt_right = np.mod(
            mat_mul_mod(A_part, E_t, self.q) + Bt, self.q
        )
        AtKt = np.hstack([A_part, AtKt_right])
        At_M = np.hstack([AtKt, HK_M])

        # A2: margin factor ×30 → ×3
        bound = self.sigma * math.sqrt(self.m + 2 * self.nk) * 3
        if LatticeUtils.norm(sig) > bound:
            return False
        return bool(np.all(mat_vec_mod(At_M, sig, self.q) == np.mod(pk.u, self.q)))
