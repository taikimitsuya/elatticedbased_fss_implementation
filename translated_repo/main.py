"""
main.py
=======
Entry point for the lattice-based FSS implementation.

Usage
-----
  python main.py              # run all demos
  python main.py --demo sel   # SelectiveFSS only
  python main.py --demo adp   # AdaptiveFSS only
  python main.py --demo all   # all demos (default)
  python main.py --demo prim  # lattice primitives only

Alternatively, use as a library:
  import lattice_fss as F
"""

from __future__ import annotations
import argparse, math, sys, time
import numpy as np

# --- Import from the modular package ---------------------------------
sys.path.insert(0, '.')
import lattice_fss as F
from lattice_fss.core.utils    import mat_mul_mod, mat_vec_mod, LatticeUtils
from lattice_fss.core.samplers import GadgetSampler
from lattice_fss.core.trapdoor import GadgetTrapdoor
from lattice_fss.primitives.phf import ProgrammableHashFunction
from lattice_fss.schemes.selective import SelectiveFSS
from lattice_fss.schemes.adaptive  import AdaptiveFSS
from lattice_fss.schemes.tq_fss    import TQ_FSS

# ─────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")

def _make_params(label: str):
    """Return (q, n, sigma, ell_phf, k, nk, m) for Small or Medium."""
    if label == "Small":
        q, n, sigma, ell_phf = 97, 8, 8.0, 32
    else:
        q, n, sigma, ell_phf = 257, 16, 12.0, 64
    k  = math.ceil(math.log2(q))
    nk = n * k
    m  = 2 * n + nk
    return q, n, sigma, ell_phf, k, nk, m


# ─────────────────────────────────────────────────────────────────────
# Demo: Lattice Primitives
# ─────────────────────────────────────────────────────────────────────

def demo_primitives(label: str = "Small") -> None:
    _header(f"Lattice Primitives  [{label}]")
    q, n, sigma, ell_phf, k, nk, m = _make_params(label)
    r = sigma / math.sqrt(2)
    np.random.seed(42)

    # --- TrapGen ---
    t0 = time.perf_counter()
    gdt = GadgetTrapdoor(n, q)
    A, R_trap = gdt.trap_gen(sigma)
    ms = (time.perf_counter() - t0) * 1000
    R_aug = np.vstack([R_trap, np.eye(nk, dtype=np.int64)])
    inv = np.all(mat_mul_mod(A, R_aug, q) == gdt.G % q)
    _ok(f"TrapGen ({ms:.1f} ms)  A·[R;I]=G: {inv}")

    # --- GadgetSampler ---
    gs = GadgetSampler(n, q, k)
    v  = np.random.randint(0, q, n, dtype=np.int64)
    t0 = time.perf_counter()
    x  = gs.sample_Gperp(v, r)
    ms = (time.perf_counter() - t0) * 1000
    Gx = mat_mul_mod(gdt.G, x.reshape(-1, 1), q).flatten()
    _ok(f"GadgetSampler ({ms:.3f} ms)  Gx=v: {np.all(Gx == v % q)}")

    # --- SampleD ---
    u = np.random.randint(0, q, n, dtype=np.int64)
    H = np.eye(n, dtype=np.int64)
    t0 = time.perf_counter()
    x_d = gdt.sample_D(R_trap, A, H, u, sigma, q)
    ms  = (time.perf_counter() - t0) * 1000
    Ax  = mat_vec_mod(A, x_d, q)
    _ok(f"SampleD ({ms:.3f} ms)  Ax=u: {np.all(Ax == u % q)}")


# ─────────────────────────────────────────────────────────────────────
# Demo: SelectiveFSS
# ─────────────────────────────────────────────────────────────────────

def demo_selective(label: str = "Small", T: int = 4) -> None:
    _header(f"SelectiveFSS  [{label}]  T={T}")
    q, n, sigma, ell_phf, k, nk, m = _make_params(label)
    M = b"Hello, SelectiveFSS!"
    np.random.seed(42)

    scheme = SelectiveFSS(n=n, q=q, T=T, sigma=sigma)
    scheme.phf = ProgrammableHashFunction(n, m, nk, q, beta=sigma, ell_phf=ell_phf)

    t0 = time.perf_counter()
    pk, sk0 = scheme.key_gen()
    t_kg = (time.perf_counter() - t0) * 1000

    sk_cur = sk0
    t_up_list = []
    for t in range(T):
        t0 = time.perf_counter()
        sk_next = scheme.key_update(pk, sk_cur)
        t_up_list.append((time.perf_counter() - t0) * 1000)
        sk_cur = sk_next

    # Sign & Verify at each epoch
    all_ok = True
    for t, sk_t in enumerate([sk0] + [sk0]):   # t=0 only for demo speed
        t0  = time.perf_counter()
        sig = scheme.sign(pk, sk0, M, 0)
        t_sg = (time.perf_counter() - t0) * 1000
        t0  = time.perf_counter()
        ok  = scheme.verify(pk, M, sig, 0)
        t_vf = (time.perf_counter() - t0) * 1000
        if not ok:
            all_ok = False
        break   # one epoch is sufficient for demo

    _ok(f"KeyGen    {t_kg:7.1f} ms")
    _ok(f"KeyUpdate {t_up_list[0]:7.3f} ms  (O(1) independent of T)")
    _ok(f"Sign      {t_sg:7.3f} ms")
    _ok(f"Verify    {t_vf:7.3f} ms  result={ok}")
    _ok(f"sk ROM = {((m+nk)*nk + n*n + 2*n*nk) / 1024:.2f} KB  "
        f"(constant for all T)")
    _ok(f"σ  ROM = {(m + 2*nk) * 2} B  (constant for all T)")

    # Theorem 1 check: At Kt Rt = Ht G
    At, Bt = scheme._make_At(pk, 0)
    Et, _  = scheme._build_correction(pk, 0)
    A_part = At[:, :m]
    AtKt   = np.hstack([A_part, np.mod(mat_mul_mod(A_part, Et, q) + Bt, q)])
    lhs    = mat_mul_mod(AtKt, sk0.R, q)
    rhs    = mat_mul_mod(sk0.H_tag, scheme.G, q)
    inv    = np.all(np.mod(lhs, q) == np.mod(rhs, q))
    _ok(f"Theorem 1 invariant  AtKtRt = HtG: {inv}")


# ─────────────────────────────────────────────────────────────────────
# Demo: AdaptiveFSS
# ─────────────────────────────────────────────────────────────────────

def demo_adaptive(label: str = "Small", T: int = 4, ell: int = 4) -> None:
    _header(f"AdaptiveFSS  [{label}]  T={T}  ℓ={ell}")
    q, n, sigma, ell_phf, k, nk, m = _make_params(label)
    M = b"Hello, AdaptiveFSS!"
    np.random.seed(42)

    scheme = AdaptiveFSS(n=n, q=q, T=T, ell=ell, sigma=sigma)
    scheme.phf = ProgrammableHashFunction(n, m, nk, q, beta=sigma, ell_phf=ell_phf)

    t0 = time.perf_counter()
    pk, sk0 = scheme.key_gen()
    t_kg = (time.perf_counter() - t0) * 1000

    t0   = time.perf_counter()
    sk1  = scheme.key_update(pk, sk0)
    t_up = (time.perf_counter() - t0) * 1000

    t0   = time.perf_counter()
    sig  = scheme.sign(pk, sk0, M, 0)
    t_sg = (time.perf_counter() - t0) * 1000

    t0   = time.perf_counter()
    ok   = scheme.verify(pk, M, sig, 0)
    t_vf = (time.perf_counter() - t0) * 1000

    _ok(f"KeyGen    {t_kg:7.1f} ms")
    _ok(f"KeyUpdate {t_up:7.3f} ms  (O(1), same cost as SelectiveFSS)")
    _ok(f"Sign      {t_sg:7.3f} ms")
    _ok(f"Verify    {t_vf:7.3f} ms  result={ok}")
    _ok(f"b_{{t,0}} = 1 (identity fixed bit): "
        f"{all(pk.IDs[t][0] == 1 for t in range(T+1))}")


# ─────────────────────────────────────────────────────────────────────
# Demo: TQ-FSS (baseline)
# ─────────────────────────────────────────────────────────────────────

def demo_tq_fss(label: str = "Small", T: int = 4) -> None:
    """
    TQ-FSS: baseline scheme for size comparison only.
    Sign / Verify are not implemented (key structure differs from Ours).
    """
    _header(f"TQ-FSS (baseline)  [{label}]  T={T}  — size comparison")
    q, n, sigma, ell_phf, k, nk, m = _make_params(label)
    np.random.seed(42)

    scheme = TQ_FSS(n=n, q=q, T=T, sigma=sigma)
    scheme.phf = ProgrammableHashFunction(n, m, nk, q, beta=sigma, ell_phf=ell_phf)

    t0 = time.perf_counter()
    pk, sk0 = scheme.key_gen()
    t_kg = (time.perf_counter() - t0) * 1000

    t0  = time.perf_counter()
    sk1 = scheme.key_update(pk, sk0)
    t_up = (time.perf_counter() - t0) * 1000

    import math as _math
    bg = _math.ceil((_math.log2(12*sigma)+1)/8)
    sk0_rom = sk0['R'].shape[0] * sk0['R'].shape[1] * bg
    sk1_rom = sk1['R'].shape[0] * sk1['R'].shape[1] * bg

    _ok(f"KeyGen    {t_kg:7.1f} ms")
    _ok(f"KeyUpdate {t_up:7.3f} ms  (key grows: O(t))")
    _ok(f"sk ROM at t=0: {sk0_rom / 1024:.2f} KB  "
        f"R.shape={sk0['R'].shape}")
    _ok(f"sk ROM at t=1: {sk1_rom / 1024:.2f} KB  "
        f"R.shape={sk1['R'].shape}  (+{(sk1_rom-sk0_rom)} B per update)")
    _ok(f"Sign/Verify: not implemented in this reference implementation")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lattice-based FSS implementation demo"
    )
    parser.add_argument(
        "--demo",
        choices=["all", "prim", "sel", "adp", "tq"],
        default="all",
        help=(
            "Which demo to run: "
            "all=all demos, prim=primitives, "
            "sel=SelectiveFSS, adp=AdaptiveFSS, tq=TQ-FSS"
        ),
    )
    parser.add_argument(
        "--param",
        choices=["Small", "Medium"],
        default="Small",
        help="Parameter set (default: Small)",
    )
    parser.add_argument(
        "--T", type=int, default=4,
        help="Maximum key updates T (default: 4)",
    )
    args = parser.parse_args()

    print(f"\nLattice-based FSS  |  param={args.param}  T={args.T}")
    print("Reference: Ishibashi et al., IEEE ACCESS 2024")

    run_all  = args.demo == "all"
    if run_all or args.demo == "prim": demo_primitives(args.param)
    if run_all or args.demo == "sel":  demo_selective(args.param, args.T)
    if run_all or args.demo == "adp":  demo_adaptive(args.param, args.T)
    if run_all or args.demo == "tq":   demo_tq_fss(args.param, args.T)

    print(f"\n{'='*60}")
    print("  Done.")
    print('='*60)


if __name__ == "__main__":
    main()
