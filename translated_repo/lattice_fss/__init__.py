"""
lattice_fss
===========
Python + C-extension implementation of:

  Ishibashi, Minematsu, Suzuki, Shikata,
  "Efficient Lattice-based Forward Secure Signature without Random Oracles,"
  IEEE ACCESS, 2024.

Public API
----------
  SelectiveFSS             -- selective FS-EUF-CMA scheme (§V-B)
  AdaptiveFSS              -- adaptive FS-EUF-CMA scheme  (§V-C)
  ProgrammableHashFunction -- Zhang et al. lattice PHF
  GadgetTrapdoor           -- MP2012 G-trapdoor primitives
  GadgetSampler            -- GadgetSampler (C-extension accelerated)
  LatticeUtils             -- utility methods
  mat_mul_mod              -- modular matrix multiplication
  mat_vec_mod              -- modular matrix-vector product
"""
from .core.utils     import mat_mul_mod, mat_vec_mod, LatticeUtils
from .core.samplers  import PeikertGaussianSampler, GadgetSampler
from .core.trapdoor  import GadgetTrapdoor, SwapTrap
from .primitives.phf import (
    EncodingFunction,
    PHFKey,
    ProgrammableHashFunction,
    AbortResistantHashFamily,
)
from .schemes.selective import PublicKeySelective, SigningKey, SelectiveFSS
from .schemes.adaptive  import PublicKeyAdaptive, SigningKeyAdaptive, AdaptiveFSS
from .schemes.tq_fss    import TQ_FSS

__all__ = [
    "SelectiveFSS", "AdaptiveFSS", "TQ_FSS",
    "ProgrammableHashFunction",
    "GadgetTrapdoor", "GadgetSampler", "SwapTrap",
    "LatticeUtils", "mat_mul_mod", "mat_vec_mod",
    "PublicKeySelective", "SigningKey",
    "PublicKeyAdaptive", "SigningKeyAdaptive",
    "EncodingFunction", "PHFKey", "AbortResistantHashFamily",
]
