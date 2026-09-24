"""
schemes/tq_fss.py
=================
TQ-FSS: HIBE-based forward-secure signature (baseline for comparison).
Reference: Tang and Qiu, ICICS 2020.

  BlockingOptimizer : helper
  TQ_FSS            : KeyGen / KeyUpdate / Sign / Verify
"""
from __future__ import annotations
import math
import numpy as np
from ..core.utils import mat_mul_mod, mat_vec_mod, LatticeUtils
from ..core.trapdoor import GadgetTrapdoor
from ..primitives.phf import EncodingFunction, ProgrammableHashFunction

class BlockingOptimizer:
    """Chatterjee-Sarkar blocking technique. Paper §V.C.3."""

    def __init__(self, ell: int, beta: int):
        self.ell = ell
        self.beta = beta
        self.ell_double_prime = math.ceil(ell / beta)

    def block_identity(self, ID_t: np.ndarray) -> List[np.ndarray]:
        blocks = [ID_t[0:1]]
        for j in range(self.ell_double_prime):
            s = j * self.beta + 1
            blocks.append(ID_t[s:s + self.beta])
        return blocks


class TQ_FSS:
    """TQ-FSS: Lattice-based FSS by Tang-Qiu (baseline for comparison). Paper §IV."""

    def __init__(self, n: int, q: int, T: int, sigma: float):
        self.n = n
        self.q = q
        self.T = T
        self.sigma = sigma
        self.gadget = GadgetTrapdoor(n, q)
        self.m = self.gadget.m
        self.nk = self.gadget.nk
        self.G = self.gadget.G
        self.H_enc = EncodingFunction(n, q)
        self.phf = ProgrammableHashFunction(n, self.m, self.nk, q, beta=sigma)

    def key_gen(self):
        A, R = self.gadget.trap_gen(self.sigma)
        A_p = [LatticeUtils.sample_uniform_matrix(self.n, self.nk, self.q)
               for _ in range(self.T + 1)]
        u = LatticeUtils.sample_uniform_matrix(self.n, 1, self.q).flatten()
        return {"A": A, "A_primes": A_p, "u": u, "K": self.phf.gen()}, {"R": R, "t": 0}

    def key_update(self, pk, sk_t):
        t = sk_t["t"]
        H_enc_t1 = self.H_enc(t + 1)
        col = np.mod(pk["A_primes"][t + 1] + mat_mul_mod(H_enc_t1, self.G, self.q), self.q)
        prev_A = sk_t.get("At", pk["A"])
        At = np.hstack([prev_A, col, col])
        H_prev = sk_t.get("H_tag", np.eye(self.n, dtype=np.int64))
        H_next = np.mod(H_prev + H_enc_t1, self.q)
        R_dummy = LatticeUtils.sample_discrete_gaussian_matrix(
            self.m, self.nk, self.sigma
        )
        return {"R": R_dummy, "H_tag": H_next, "At": At, "t": t + 1}
