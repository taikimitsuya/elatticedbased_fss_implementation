# Parameter Sets

## Overview

All parameter sets satisfy **Lemma 2** of Ishibashi et al. (IEEE ACCESS 2024):

```
m = 2n + nk,   k = ⌈log₂ q⌉
```

| Name   | q   | n  | k | nk  | m   | σ    | r = σ/√2 | ℓ_φ |
|--------|-----|----|----|-----|-----|------|-----------|-----|
| Small  | 97  | 8  | 7 | 56  | 72  | 8.0  | 5.657     | 32  |
| Medium | 257 | 16 | 9 | 144 | 176 | 12.0 | 8.485     | 64  |

## Key and Signature Sizes (T = 1,000 key updates)

### SelectiveFSS

| Parameter | sk ROM  | sk RAM  | σ ROM | σ RAM  | pk ROM   |
|-----------|---------|---------|-------|--------|----------|
| Small     | 7.94 KB | 63.5 KB | 368 B | 1,472 B| 452.1 KB |
| Medium    | 99.5 KB | 398 KB  | 928 B | 3,712 B| 4,793 KB |

- **sk and σ are constant in T** (O(1) vs. O(t) for TQ-FSS).
- pk grows linearly in T due to the {A'_t} family.

### TQ-FSS (baseline, Small parameters)

| t       | sk ROM      | σ ROM     |
|---------|-------------|-----------|
| 10      | 65.2 KB     | 1,304 B   |
| 100     | 616 KB      | 11,088 B  |
| 1,000   | 6,128.9 KB  | 112,184 B |

**Reduction at t = 1,000:** sk 1/772×, σ 1/305×

## ROM Size Formulas

```
pk ROM = (n·m + n + T·n·nk + ℓ_φ·n·nk) × bpe_q
sk ROM = (m+nk)·nk·bpe_g + n²·bpe_q + 2n·nk·bpe_g
σ  ROM = (m + 2nk) × bpe_s

bpe_q = ⌈log₂(q) / 8⌉          bytes per Z_q element
bpe_g = ⌈(log₂(12σ) + 1) / 8⌉  bytes per Gaussian sample
bpe_s = ⌈(log₂(2σ√(m+2nk)) + 1) / 8⌉  bytes per σ coefficient
```

## IoT Deployment Note

For epoch t, `Sign` requires only:
- A, u, K, A'_0, A'_t

Minimum resident storage: ~15.5 KB ROM (Small, T=1,000),
reduced from 452 KB for the full pk.
The remaining {A'_t'} can be fetched from a server at key-update time.
