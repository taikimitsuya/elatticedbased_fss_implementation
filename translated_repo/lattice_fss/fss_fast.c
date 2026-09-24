/*
 * fss_fast.c  —  C extension library for lattice-based FSS acceleration
 *
 * Compilation:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC \
 *       -o fss_fast.so fss_fast.c
 *
 * Exported functions:
 *   1. sample_z_batch      : 1-D discrete Gaussian sampling (rejection sampling)
 *   2. sample_gperp        : C implementation of GadgetSampler.sample_Gperp
 *   3. mat_mul_mod_int64   : (n×m) @ (m×p) mod q — int64 matrix multiplication
 *   4. mat_vec_mod_int64   : (n×m) @ (m,)   mod q — int64 matrix-vector product
 */

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ── xoshiro256** PRNG (fast, high statistical quality) ────────── */
static uint64_t xo_state[4] = {
    0x123456789abcdef0ULL, 0xfedcba9876543210ULL,
    0xdeadbeefcafebabeULL, 0x0123456789abcdefULL
};

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xo_next(void) {
    const uint64_t result = rotl(xo_state[1] * 5, 7) * 9;
    const uint64_t t = xo_state[1] << 17;
    xo_state[2] ^= xo_state[0];
    xo_state[3] ^= xo_state[1];
    xo_state[1] ^= xo_state[2];
    xo_state[0] ^= xo_state[3];
    xo_state[2] ^= t;
    xo_state[3] = rotl(xo_state[3], 45);
    return result;
}

/* Set PRNG seed */
void fss_seed(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    xo_state[0] = s0; xo_state[1] = s1;
    xo_state[2] = s2; xo_state[3] = s3;
}

/* Uniform double in [0, 1) */
static inline double xo_double(void) {
    return (xo_next() >> 11) * (1.0 / (1ULL << 53));
}

/* Uniform integer in [lo, hi] */
static inline int64_t xo_range(int64_t lo, int64_t hi) {
    uint64_t range = (uint64_t)(hi - lo + 1);
    return lo + (int64_t)(xo_next() % range);
}

/* ── 1. 1-D discrete Gaussian sampler ─────────────────────────────
 *
 * Returns a sample from D_{Z, sigma, center} via rejection sampling.
 * tail_cut: truncates the range |k - center| > tail_cut * sigma.
 * 
 * parity: 0=even only, 1=odd only, -1=no constraint
 */
int64_t sample_z(double center, double sigma, int tail_cut, int parity) {
    int64_t lo = (int64_t)floor(center - tail_cut * sigma);
    int64_t hi = (int64_t)ceil (center + tail_cut * sigma);
    const double inv_sigma2 = M_PI / (sigma * sigma);

    while (1) {
        int64_t k = xo_range(lo, hi);
        /* Parity constraint */
        if (parity >= 0 && (int)(k % 2 + 2) % 2 != parity) continue;
        double d = (double)(k) - center;
        double rho = exp(-inv_sigma2 * d * d);
        if (xo_double() < rho) return k;
    }
}

/* ── 2. C implementation of GadgetSampler.sample_Gperp ────────────
 *
 * G = I_n ⊗ g^T,  g = (1, 2, 4, ..., 2^{k-1})
 * Returns a short x ∈ Z^{nk} satisfying Gx = v mod q.
 *
 * Output: written to out[nk] (pre-allocated by caller)
 */
void sample_gperp(
    const int64_t* v,   /* input target v ∈ Z_q^n, length n */
    int64_t* out,       /* output x ∈ Z^{nk} */
    int n, int q, int k,
    double r,           /* Gaussian parameter */
    int tail_cut        /* tail-cut factor (typically 6) */
) {
    for (int i = 0; i < n; i++) {
        int64_t carry = ((int64_t)v[i] % q + q) % q;
        for (int j = 0; j < k - 1; j++) {
            int parity = (int)(((carry % 2) + 2) % 2);
            int64_t xij = sample_z(0.0, r, tail_cut, parity);
            out[i * k + j] = xij;
            carry = (carry - xij) >> 1;
        }
        out[i * k + (k - 1)] = carry;
    }
}

/* ── 3. int64 matrix multiplication mod q: C = A @ B mod q ────────
 *
 * A: (rows_A × cols_A), B: (cols_A × cols_B)
 * C: (rows_A × cols_B)  ← pre-allocated by caller
 *
 * Maximum inner product in int64 = (q-1)^2 * cols_A
 * q < 2^20, cols_A < 2^12 → max ≈ 2^52 → safe within int64
 */
void mat_mul_mod_int64(
    const int64_t* A, const int64_t* B, int64_t* C,
    int rows_A, int cols_A, int cols_B, int64_t q
) {
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            int64_t acc = 0;
            for (int l = 0; l < cols_A; l++) {
                acc += A[i * cols_A + l] * B[l * cols_B + j];
            }
            /* acc is at most (q-1)^2 * cols_A — within int64 range */
            C[i * cols_B + j] = ((acc % q) + q) % q;
        }
    }
}

/* ── 4. int64 matrix-vector product mod q: y = A @ x mod q ────────
 *
 * A: (rows × cols), x: (cols,), y: (rows,)
 */
void mat_vec_mod_int64(
    const int64_t* A, const int64_t* x, int64_t* y,
    int rows, int cols, int64_t q
) {
    for (int i = 0; i < rows; i++) {
        int64_t acc = 0;
        for (int j = 0; j < cols; j++) {
            acc += A[i * cols + j] * x[j];
        }
        y[i] = ((acc % q) + q) % q;
    }
}
