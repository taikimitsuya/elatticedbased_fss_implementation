"""
benchmark_fss.py  —  Lattice-based FSS benchmark (sequential sampling)
════════════════════════════════════════════════════════════════════════════════

[Trial count policy]

  The conventional approach of dividing TARGET_SEC by one execution time
  provides statistically insufficient trial counts for high-CV operations
  (GadgetSampler CV≈41%, SampleD CV≈29%).

  We adopt "Sequential Sampling".

  ─── Algorithm ──────────────────────────────────────────────────────
  Step 1  Pilot measurement: run N_PILOT = 30 times and estimate CV.
  Step 2  Compute required trial count from statistical theory:
            N_req = ceil( (t_{N_pilot-1, α/2}  ×  CV  /  (REL_PREC/2))^2 )
          where REL_PREC = target relative precision (default 5%).
          This is the N guaranteeing "95% CI width ÷ mean ≤ REL_PREC".
  Step 3  Measure additional (N_req - N_pilot) runs.
          Stop early if MAX_SEC (default 60 s) is exceeded,
          and record the actually achieved relative precision.
  Step 4  Compute statistics (mean, median, SD, CV, CI, min, max) over all samples.
  ─────────────────────────────────────────────────────────────────────

  This approach automatically optimizes per operation:
    - TrapGen (CV=4%)       → N ≈ 12 suffices (avoids over-measurement)
    - GadgetSampler (CV=41%) → measures up to N ≈ 1039 (ensures precision)

  Parameters:
    N_PILOT    = 30       pilot trial count
    N_MIN      = 30       minimum guaranteed trial count
    N_MAX      = 2000     maximum trial count
    REL_PREC   = 0.05     target relative precision (CI width / mean)
    CI_LEVEL   = 0.95     confidence level
    MAX_SEC    = 60.0     max measurement time per operation (seconds)
    WARMUP     = 3        number of warm-up runs
"""

from __future__ import annotations

import math
import time
import statistics
import sys
import os
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Tuple, Optional

import numpy as np
from scipy import stats as sp_stats

# ── Module resolution (Google Colab / Jupyter compatible) ──────────────
def _resolve_module_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

sys.path.insert(0, _resolve_module_dir())
from lattice_fss import (
    GadgetTrapdoor, GadgetSampler, SwapTrap, LatticeUtils,
    SelectiveFSS, AdaptiveFSS, EncodingFunction,
    ProgrammableHashFunction, mat_mul_mod, _FAST_AVAILABLE
)

# ════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════
N_PILOT   = 100      # pilot trial count (increased from 30→100 for outlier robustness)
N_MIN     = 100      # minimum guaranteed trial count
N_MAX     = 2000     # upper limit
REL_PREC  = 0.05     # target relative precision (CI width / mean ≤ 5%)
CI_LEVEL  = 0.95     # confidence level
MAX_SEC   = 120.0    # max measurement time per operation (seconds): 60→120
WARMUP    = 5        # warm-up runs: 3→5 (for cache stabilization)
TRIM_PCT  = 0.05     # trim fraction (exclude top/bottom 5% as outliers for CV estimation)


# ════════════════════════════════════════════════════════════════
# Sequential sampling engine
# ════════════════════════════════════════════════════════════════
@dataclass
class BenchResult:
    name:              str
    n_trials:          int
    times_ms:          List[float]
    n_required:        int          # theoretically required N
    achieved_rel_prec: float        # actually achieved relative precision
    time_limited:      bool         # whether measurement was stopped by time limit

    # ── Statistics ───────────────────────────────────────────────
    @property
    def mean(self)   -> float: return statistics.mean(self.times_ms)
    @property
    def median(self) -> float: return statistics.median(self.times_ms)
    @property
    def stdev(self)  -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
    @property
    def minimum(self) -> float: return min(self.times_ms)
    @property
    def maximum(self) -> float: return max(self.times_ms)
    @property
    def cv(self) -> float:
        return (self.stdev / self.mean * 100) if self.mean > 0 else 0.0

    def confidence_interval(self) -> Tuple[float, float]:
        n = len(self.times_ms)
        if n < 2: return (self.mean, self.mean)
        se    = self.stdev / math.sqrt(n)
        t_val = sp_stats.t.ppf((1 + CI_LEVEL) / 2, df=n - 1)
        return (self.mean - t_val * se, self.mean + t_val * se)

    def summary_line(self) -> str:
        lo, hi = self.confidence_interval()
        prec_str = f"{self.achieved_rel_prec*100:.1f}%"
            flag = "⏱ time-limited" if self.time_limited else "✓"
        cv_trim_str = f"{_trimmed_cv(self.times_ms):.1f}%"
        return (
            f"  mean={self.mean:9.4f}ms  median={self.median:9.4f}ms"
            f"  SD={self.stdev:8.4f}"
            f"  CV={self.cv:.1f}%(overall) CV_trim={cv_trim_str}(outliers excluded)"
            f"  95%CI=[{lo:.4f},{hi:.4f}]"
            f"  min={self.minimum:.4f}  max={self.maximum:.4f}"
            f"  N={self.n_trials}(req={self.n_required})"
            f"  achieved precision={prec_str} {flag}"
        )


def _run_once(fn: Callable) -> float:
    """Execute fn once and return elapsed time [ms]."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def _trimmed_cv(times: List[float], trim_pct: float = TRIM_PCT) -> float:
    """
    Robust CV estimation based on trimmed standard deviation.

    Sporadic outliers from OS scheduling noise (rare runs several times longer
    than normal) can significantly distort the standard CV.
    Computing CV on the sample after trimming the top/bottom trim_pct%
    eliminates this effect.

    Final statistics (mean, CI, etc.) are computed over the full raw data.
    Trimming is used only for "N_req estimation".

    Example: SwapTrap
      Raw data CV = 16.2% (including outliers)
      Trimmed CV  =  2.2% (core portion only)
    """
    if len(times) < 4:
        import statistics as _st
        mn = _st.mean(times)
        sd = _st.stdev(times) if len(times) > 1 else 0.0
        return (sd / mn * 100) if mn > 0 else 0.0
    arr  = np.array(times, dtype=float)
    lo   = np.percentile(arr, trim_pct * 100)
    hi   = np.percentile(arr, (1 - trim_pct) * 100)
    core = arr[(arr >= lo) & (arr <= hi)]
    if len(core) < 2 or core.mean() == 0:
        return float(arr.std() / arr.mean() * 100) if arr.mean() > 0 else 0.0
    return float(core.std() / core.mean() * 100)


def _calc_n_required(cv_pct: float, n_pilot: int) -> int:
    """
    Compute the required trial count to achieve target relative precision REL_PREC.

    N = ceil( (t_{n_pilot-1, 0.975} x (CV/100) / (REL_PREC/2))^2 )

    Uses trimmed CV (coefficient of variation after outlier removal)
    to prevent OS-noise outliers from distorting N_req.
    """
    if cv_pct <= 0:
        return N_MIN
    t_val = sp_stats.t.ppf((1 + CI_LEVEL) / 2, df=max(n_pilot - 1, 1))
    n_req = math.ceil((t_val * (cv_pct / 100) / (REL_PREC / 2)) ** 2)
    return max(N_MIN, min(N_MAX, n_req))


def measure(label: str, fn: Callable) -> BenchResult:
    """
    Measure execution time of fn using sequential sampling.

    Step 1: Warm-up (cache / JIT stabilization)
    Step 2: Pilot measurement N_PILOT runs → estimate CV
    Step 3: Compute N_req from target precision
    Step 4: Additional measurements (within time limit MAX_SEC)
    Step 5: Return statistics
    """
    # Step 1: Warm-up
    for _ in range(WARMUP):
        fn()

    # Step 2: Pilot measurement
    pilot_times: List[float] = [_run_once(fn) for _ in range(N_PILOT)]
    # Compute N_req with outlier influence removed via trimmed CV
    pilot_cv_trim = _trimmed_cv(pilot_times)
    pilot_cv_raw  = (statistics.stdev(pilot_times) / statistics.mean(pilot_times) * 100
                     if len(pilot_times) > 1 else 0.0)

    # Step 3: Compute required trial count (trimmed CV-based)
    n_req = _calc_n_required(pilot_cv_trim, N_PILOT)

    # Step 4: Additional measurements
    all_times: List[float] = pilot_times.copy()
    deadline = time.perf_counter() + MAX_SEC
    time_limited = False
    n_add = n_req - N_PILOT

    for _ in range(max(0, n_add)):
        if time.perf_counter() > deadline:
            time_limited = True
            break
        all_times.append(_run_once(fn))

    # Step 5: Compute achieved precision
    n_actual = len(all_times)
    sd_actual = statistics.stdev(all_times) if n_actual > 1 else 0.0
    mean_actual = statistics.mean(all_times)
    t_val = sp_stats.t.ppf((1 + CI_LEVEL) / 2, df=max(n_actual - 1, 1))
    achieved = (2 * t_val * (sd_actual / mean_actual) / math.sqrt(n_actual)
                if mean_actual > 0 else float('inf'))

    return BenchResult(
        name=label,
        n_trials=n_actual,
        times_ms=all_times,
        n_required=n_req,
        achieved_rel_prec=achieved,
        time_limited=time_limited,
    )


# ════════════════════════════════════════════════════════════════
# Parameter sets
# ════════════════════════════════════════════════════════════════
@dataclass
class ParamSet:
    name:    str
    q:       int
    n:       int
    sigma:   float
    ell_phf: int
    T:       int
    ell_id:  int
    k:    int = field(init=False)
    nk:   int = field(init=False)
    m:    int = field(init=False)

    def __post_init__(self):
        self.k  = math.ceil(math.log2(self.q))
        self.nk = self.n * self.k
        self.m  = 2 * self.n + self.nk

    def lemma2_check(self) -> Tuple[bool, float, float]:
        need = (self.n + 1) * math.log2(self.q) + 2 * math.log2(max(self.n, 2))
        return self.m > need, float(self.m), need


PARAM_SETS = [
    ParamSet(name="Small",  q=97,  n=8,  sigma=8.0,  ell_phf=32, T=4, ell_id=4),
    ParamSet(name="Medium", q=257, n=16, sigma=12.0, ell_phf=64, T=4, ell_id=4),
]


# ════════════════════════════════════════════════════════════════
# Benchmark groups
# ════════════════════════════════════════════════════════════════

def bench_primitives(p: ParamSet) -> Dict[str, BenchResult]:
    np.random.seed(42)
    gdt = GadgetTrapdoor(p.n, p.q)
    A, R = gdt.trap_gen(p.sigma)
    r   = p.sigma / math.sqrt(2.0)
    gs  = GadgetSampler(p.n, p.q, p.k)
    H   = np.eye(p.n, dtype=np.int64)
    u   = np.random.randint(0, p.q, p.n, dtype=np.int64)
    v   = np.random.randint(0, p.q, p.n, dtype=np.int64)

    return {
        "TrapGen":
            measure("TrapGen",  lambda: gdt.trap_gen(p.sigma)),
        "SampleD":
            measure("SampleD",  lambda: gdt.sample_D(R, A, H, u, p.sigma, p.q)),
        "GadgetSampler.sample_Gperp":
            measure("GadgetSampler.sample_Gperp", lambda: gs.sample_Gperp(v, r)),
    }


def bench_swaptrap(p: ParamSet) -> Dict[str, BenchResult]:
    np.random.seed(42)
    gdt = GadgetTrapdoor(p.n, p.q)
    r   = p.sigma / math.sqrt(2.0)
    sw  = SwapTrap(gdt, p.q, r)
    A, R_trap = gdt.trap_gen(p.sigma)
    H_enc = EncodingFunction(p.n, p.q)
    Ap = [LatticeUtils.sample_uniform_matrix(p.n, p.nk, p.q) for _ in range(2)]
    B0 = np.mod(Ap[0] + mat_mul_mod(H_enc(0), gdt.G, p.q), p.q)
    A0 = np.hstack([A, B0])
    sel_tmp = SelectiveFSS(n=p.n, q=p.q, T=2, sigma=p.sigma)
    H0 = sel_tmp._crh(np.eye(p.n, dtype=np.int64), 0)
    H1 = sel_tmp._crh(H0, 1)
    A0_ext = np.hstack([A0, mat_mul_mod(H0, gdt.G, p.q)])
    R0 = gdt.del_trap(A0_ext, R_trap, np.eye(p.n, dtype=np.int64), H0, p.sigma, p.q)

    return {
        "SwapTrap.execute":
            measure("SwapTrap.execute", lambda: sw.execute(A0, R0, H0, H1, p.sigma)),
    }


def _make_sel(p: ParamSet) -> SelectiveFSS:
    s = SelectiveFSS(n=p.n, q=p.q, T=p.T, sigma=p.sigma)
    s.phf = ProgrammableHashFunction(p.n, p.m, p.nk, p.q,
                                     beta=p.sigma, ell_phf=p.ell_phf)
    return s


def _make_adp(p: ParamSet) -> AdaptiveFSS:
    s = AdaptiveFSS(n=p.n, q=p.q, T=p.T, ell=p.ell_id, sigma=p.sigma)
    s.phf = ProgrammableHashFunction(p.n, p.m, p.nk, p.q,
                                     beta=p.sigma, ell_phf=p.ell_phf)
    return s


def bench_selective(p: ParamSet) -> Dict[str, BenchResult]:
    np.random.seed(42)
    M = b"benchmark_selective"
    scheme = _make_sel(p)
    pk, sk0 = scheme.key_gen()
    sig0 = scheme.sign(pk, sk0, M, 0)

    return {
        "SelectiveFSS.KeyGen":
            measure("SelectiveFSS.KeyGen",    lambda: _make_sel(p).key_gen()),
        "SelectiveFSS.KeyUpdate":
            measure("SelectiveFSS.KeyUpdate", lambda: scheme.key_update(pk, sk0)),
        "SelectiveFSS.Sign":
            measure("SelectiveFSS.Sign",      lambda: scheme.sign(pk, sk0, M, 0)),
        "SelectiveFSS.Verify":
            measure("SelectiveFSS.Verify",    lambda: scheme.verify(pk, M, sig0, 0)),
    }


def bench_adaptive(p: ParamSet) -> Dict[str, BenchResult]:
    np.random.seed(42)
    M = b"benchmark_adaptive"
    scheme = _make_adp(p)
    pk2, sk2 = scheme.key_gen()
    sig2 = scheme.sign(pk2, sk2, M, 0)

    return {
        "AdaptiveFSS.KeyGen":
            measure("AdaptiveFSS.KeyGen",    lambda: _make_adp(p).key_gen()),
        "AdaptiveFSS.KeyUpdate":
            measure("AdaptiveFSS.KeyUpdate", lambda: scheme.key_update(pk2, sk2)),
        "AdaptiveFSS.Sign":
            measure("AdaptiveFSS.Sign",      lambda: scheme.sign(pk2, sk2, M, 0)),
        "AdaptiveFSS.Verify":
            measure("AdaptiveFSS.Verify",    lambda: scheme.verify(pk2, M, sig2, 0)),
    }


# ════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════

def _sep(char="═", w=100): print(char * w)


def _param_info(p: ParamSet):
    ok, m_val, need = p.lemma2_check()
    print(f"  q={p.q}, n={p.n}, k={p.k}, nk={p.nk}, m={p.m}, sigma={p.sigma}")
    print(f"  ell_phf={p.ell_phf}, T={p.T}, ell_id={p.ell_id}")
    print(f"  Lemma 2: m={int(m_val)} > {need:.1f} → {'OK ✓' if ok else 'NG ✗'}")
    print(f"  C extension: {'enabled ✓' if _FAST_AVAILABLE else 'disabled (fallback)'}")    
    print(f"  Signature size |σ| = m+2nk = {p.m + 2*p.nk}")


def run_benchmark(p: ParamSet) -> Dict[str, BenchResult]:
    _sep()
    print(f"  Parameter set: {p.name}")
    _sep()
    _param_info(p)
    print()

    all_res: Dict[str, BenchResult] = {}
    groups = [
        ("Lattice primitives", bench_primitives),
        ("SwapTrap",        bench_swaptrap),
        ("SelectiveFSS",    bench_selective),
        ("AdaptiveFSS",     bench_adaptive),
    ]

    for gname, fn in groups:
        print(f"  ── {gname} ──")
        t0 = time.perf_counter()
        try:
            res = fn(p)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue
        elapsed = time.perf_counter() - t0

        for label, r in res.items():
            prec_mark = ("✓" if r.achieved_rel_prec <= REL_PREC
                         else f"△ {r.achieved_rel_prec*100:.1f}%")
            print(f"  {label}")
            print(r.summary_line())
            print(f"    → Precision target ({REL_PREC*100:.0f}%): {prec_mark}"
                  f"  {'⏱time-limited' if r.time_limited else ''}")    
        print(f"  (Group measurement time: {elapsed:.1f}s)\n")
        all_res.update(res)

    return all_res


def _cmp_table(all_param: Dict[str, Dict[str, BenchResult]]):
    """Print cross-parameter comparison table."""
    _sep("─")
    print("  Comparison table (mean [ms] / N / achieved precision)")
    _sep("─")
    ops = list(next(iter(all_param.values())).keys())
    pnames = list(all_param.keys())
    W = 16
    print(f"  {'Operation':38s}" + "".join(f"{n:>{W}}" for n in pnames))
    print("  " + "─" * (38 + W * len(pnames)))
    for op in ops:
        row = f"  {op:38s}"
        for pn in pnames:
            rd = all_param.get(pn, {})
            if op in rd:
                r = rd[op]
                prec = r.achieved_rel_prec * 100
                flag=' ⏱' if r.time_limited else '  '; row += f"{r.mean:>8.3f}ms N={r.n_trials:<4d}({prec:.1f}%) {flag}"
            else:
                row += f"{'N/A':>{W}}"
        print(row)
    _sep("─")


def _n_table(all_param: Dict[str, Dict[str, BenchResult]]):
    """Print trial count table (target vs actual)."""
    _sep("─")
    print("  Trial count table [actual N (required N_req)]")
    print(f"  Target relative precision: {REL_PREC*100:.0f}%  |  N_PILOT={N_PILOT}  |  N_MAX={N_MAX}")
    _sep("─")
    ops = list(next(iter(all_param.values())).keys())
    pnames = list(all_param.keys())
    W = 22
    print(f"  {'Operation':38s}" + "".join(f"{n:>{W}}" for n in pnames))
    print("  " + "─" * (38 + W * len(pnames)))
    for op in ops:
        row = f"  {op:38s}"
        for pn in pnames:
            rd = all_param.get(pn, {})
            if op in rd:
                r = rd[op]
                flag = "⏱" if r.time_limited else " "
                row += f"  N={r.n_trials:>5d}(req={r.n_required:>5d}){flag:>{W-20}}"
            else:
                row += f"{'N/A':>{W}}"
        print(row)
    _sep("─")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    _sep("═")
    print("  Lattice-based FSS benchmark (sequential sampling)")
    print("  Paper: Ishibashi et al., IEEE ACCESS (2024)")
    _sep("═")
    print(f"  Measurement settings:")
    print(f"    Pilot runs      : {N_PILOT}")
    print(f"    Minimum runs    : {N_MIN}")
    print(f"    Maximum runs    : {N_MAX}")
    print(f"    Target precision: {REL_PREC*100:.0f}%  (CI width / mean ≤ {REL_PREC*100:.0f}%)")
    print(f"    CI level        : {CI_LEVEL*100:.0f}% (t-distribution)")
    print(f"    Time limit/op   : {MAX_SEC:.0f} s")
    print(f"    Warm-up runs    : {WARMUP}")
    print(f"    Trim fraction   : {TRIM_PCT*100:.0f}% (outlier removal for CV estimation)")
    print(f"    CV estimation   : trimmed std dev (outliers excluded from N_req)")
    print(f"    C extension     : {'enabled (fss_fast.so)' if _FAST_AVAILABLE else 'disabled'}")    
    print()

    t_total = time.perf_counter()
    all_param: Dict[str, Dict[str, BenchResult]] = {}

    for p in PARAM_SETS:
        all_param[p.name] = run_benchmark(p)

    _sep("═")
    print("  Summary")
    _sep("═")
    _cmp_table(all_param)
    print()
    _n_table(all_param)
    print()
    print(f"  Total measurement time: {time.perf_counter()-t_total:.1f} s")
    _sep("═")


if __name__ == "__main__":
    main()
