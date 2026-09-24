"""
core/utils.py
=============
Lattice utility functions and matrix operations.

  _make_psd          : positive semi-definite matrix helper
  _mod_centered      : centered modular reduction
  _load_fast_lib     : C extension loader
  mat_mul_mod        : modular matrix multiplication (C / NumPy / Python)
  mat_vec_mod        : modular matrix-vector product
  LatticeUtils       : static lattice utility methods
"""
from __future__ import annotations
import math, ctypes, hashlib, os
from typing import Optional
import os as _os
import secrets
import numpy as np
from scipy.linalg import cholesky

def _make_psd(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Restores positive definiteness of a matrix that became
    indefinite due to floating-point errors, via minimum eigenvalue shift.
    """
    M = (M + M.T) / 2
    lam_min = np.linalg.eigvalsh(M).min()
    if lam_min < eps:
        M += (eps - lam_min) * np.eye(M.shape[0])
    return M


def _mod_centered(x: np.ndarray, q: int) -> np.ndarray:
    """Returns x mod q centered in (-q/2, q/2] (integer array)."""
    x = np.mod(x, q)
    mask = x > q // 2
    x[mask] -= q
    return x


def _load_fast_lib():
    """Search for and load fss_fast.so. Returns None if not found."""
    candidates = [
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'fss_fast.so'),
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'fss_fast.so'),
        _os.path.join(_os.getcwd(), 'fss_fast.so'),
        '/tmp/fss_fast.so',
    ]
    for path in candidates:
        if _os.path.exists(path):
            try:
                lib = _ctypes.CDLL(path)
                # Set function signatures
                lib.sample_gperp.argtypes = [
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.c_int, _ctypes.c_int, _ctypes.c_int,
                    _ctypes.c_double, _ctypes.c_int,
                ]
                lib.sample_gperp.restype = None
                lib.mat_mul_mod_int64.argtypes = [
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.c_int, _ctypes.c_int, _ctypes.c_int, _ctypes.c_int64,
                ]
                lib.mat_mul_mod_int64.restype = None
                lib.mat_vec_mod_int64.argtypes = [
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.POINTER(_ctypes.c_int64),
                    _ctypes.c_int, _ctypes.c_int, _ctypes.c_int64,
                ]
                lib.mat_vec_mod_int64.restype = None
                # Initialize PRNG seed
                try:
                    lib.fss_seed.argtypes = [_ctypes.c_uint64]*4
                    lib.fss_seed.restype  = None
                    lib.fss_seed(0x1234567890abcdef, 0xfedcba0987654321,
                                 0xdeadbeefcafebabe, 0x0123456789abcdef)
                except Exception:
                    pass
                return lib
            except Exception:
                pass
    return None

# Load C extension at import time
_FAST_LIB       = _load_fast_lib()
_FAST_AVAILABLE = _FAST_LIB is not None
_fast_lib       = _FAST_LIB  # alias


def mat_mul_mod(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
    """
    A @ B mod q  — 3-tier fallback.

    tier-1: C extension (fss_fast.so) — SIMD/O3 optimized
    tier-2: NumPy int64 @ — BLAS optimized, safe within int64 range
    tier-3: Python object loop — full overflow protection
    """
    A = np.ascontiguousarray(A, dtype=np.int64)
    B = np.ascontiguousarray(B, dtype=np.int64)

    if _FAST_AVAILABLE:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
        _FAST_LIB.mat_mul_mod_int64(
            A.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
            B.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
            C.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
            A.shape[0], A.shape[1], B.shape[1], q,
        )
        return C

    # tier-2: NumPy @ (overflow safety verified)
    max_inner = int(q - 1) ** 2 * A.shape[1]
    if max_inner < (2**62):
        return np.mod(A @ B, q)

    # tier-3: fallback
    rows, cols = A.shape[0], B.shape[1]
    out = np.empty((rows, cols), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = int(np.dot(A[i].astype(object), B[:, j].astype(object))) % q
    return out


def mat_vec_mod(A: np.ndarray, v: np.ndarray, q: int) -> np.ndarray:
    """
    A @ v mod q  — 3-tier fallback.
    """
    A = np.ascontiguousarray(A, dtype=np.int64)
    v = np.ascontiguousarray(v, dtype=np.int64)

    if _FAST_AVAILABLE:
        y = np.zeros(A.shape[0], dtype=np.int64)
        _FAST_LIB.mat_vec_mod_int64(
            A.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
            v.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
            y.ctypes.data_as(_ctypes.POINTER(_ctypes.c_int64)),
            A.shape[0], A.shape[1], q,
        )
        return y

    max_inner = int(q - 1) ** 2 * A.shape[1]
    if max_inner < (2**62):
        return np.mod(A @ v, q)

    out = np.empty(A.shape[0], dtype=np.int64)
    for i in range(A.shape[0]):
        out[i] = int(np.dot(A[i].astype(object), v.astype(object))) % q
    return out


class LatticeUtils:
    """Lattice arithmetic utility methods."""

    @staticmethod
    def mod_matrix(M: np.ndarray, q: int) -> np.ndarray:
        return np.mod(M, q)

    @staticmethod
    def sample_uniform_matrix(rows: int, cols: int, q: int) -> np.ndarray:
        return np.array(
            [[secrets.randbelow(q) for _ in range(cols)] for _ in range(rows)],
            dtype=np.int64,
        )

    @staticmethod
    def sample_discrete_gaussian_vector(dim: int, sigma: float) -> np.ndarray:
        """D_{Z^dim, sigma, 0}: precise discrete Gaussian sampling via rejection sampling."""
        from .samplers import PeikertGaussianSampler as _PGS
        return np.array(
            [_PGS.sample_z(0.0, sigma) for _ in range(dim)],
            dtype=np.int64,
        )

    @staticmethod
    def sample_discrete_gaussian_matrix(rows: int, cols: int, sigma: float) -> np.ndarray:
        """D_{Z^{rows×cols}, sigma, 0}: element-wise rejection sampling."""
        out = np.empty((rows, cols), dtype=np.int64)
        from .samplers import PeikertGaussianSampler as _PGS
        for i in range(rows):
            for j in range(cols):
                out[i, j] = _PGS.sample_z(0.0, sigma)
        return out

    @staticmethod
    def largest_singular_value(R: np.ndarray) -> float:
        return float(np.linalg.norm(R, ord=2))

    @staticmethod
    def norm(v: np.ndarray) -> float:
        return float(np.linalg.norm(v))

    @staticmethod
    def mat_mul_mod(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
        return mat_mul_mod(A, B, q)

    @staticmethod
    def mat_vec_mod(A: np.ndarray, v: np.ndarray, q: int) -> np.ndarray:
        return mat_vec_mod(A, v, q)

    @staticmethod
    def inv_mod(A: np.ndarray, q: int) -> np.ndarray:
        """
        Inverse of A over Z_q using Gauss-Jordan elimination mod q.
        Raises ValueError if A is not invertible.
        """
        n = A.shape[0]
        aug = np.hstack([A.copy().astype(np.int64), np.eye(n, dtype=np.int64)])
        for col in range(n):
            pivot = next(
                (row for row in range(col, n) if aug[row, col] % q != 0), -1
            )
            if pivot == -1:
                raise ValueError("Matrix is not invertible over Z_q.")
            aug[[col, pivot]] = aug[[pivot, col]]
            piv_inv = pow(int(aug[col, col]) % q, -1, q)
            aug[col] = (aug[col] * piv_inv) % q
            for row in range(n):
                if row != col and aug[row, col] % q != 0:
                    aug[row] = (aug[row] - int(aug[row, col]) * aug[col]) % q
        return aug[:, n:]
