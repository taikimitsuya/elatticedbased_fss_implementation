"""
Correctness tests for the lattice-based FSS implementation.
Run:  python -m pytest tests/test_correctness.py -v
"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
import pytest
import lattice_fss as F

SEED = 42
M    = b"test_message"


def make_sel(q, n, sigma, ell_phf, T=4):
    k  = math.ceil(math.log2(q))
    nk = n * k
    m  = 2 * n + nk
    sel = F.SelectiveFSS(n=n, q=q, T=T, sigma=sigma)
    sel.phf = F.ProgrammableHashFunction(n, m, nk, q, beta=sigma, ell_phf=ell_phf)
    return sel, k, nk, m


@pytest.fixture(params=[
    ("Small",  97,  8,  8.0, 32),
    ("Medium", 257, 16, 12.0, 64),
], ids=["Small", "Medium"])
def sel_fixture(request):
    np.random.seed(SEED)
    _, q, n, sigma, ell_phf = request.param
    sel, k, nk, m = make_sel(q, n, sigma, ell_phf)
    pk, sk0 = sel.key_gen()
    sks = [sk0]
    for _ in range(3):
        sks.append(sel.key_update(pk, sks[-1]))
    return sel, pk, sks, q, n, k, nk, m


class TestTrapGen:
    def test_invariant(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        gdt  = F.GadgetTrapdoor(n, q)
        A, R = gdt.trap_gen(sel.sigma)
        aug  = np.vstack([R, np.eye(nk, dtype=np.int64)])
        assert np.all(F.mat_mul_mod(A, aug, q) == gdt.G % q)


class TestTheorem1:
    def test_all_t(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        for t, sk_t in enumerate(sks):
            At, Bt = sel._make_At(pk, t)
            Et, _  = sel._build_correction(pk, t)
            A_p    = At[:, :m]
            AtKt   = np.hstack([A_p,
                                 np.mod(F.mat_mul_mod(A_p, Et, q) + Bt, q)])
            lhs = F.mat_mul_mod(AtKt, sk_t.R, q)
            rhs = F.mat_mul_mod(sk_t.H_tag, sel.G, q)
            assert np.all(np.mod(lhs, q) == np.mod(rhs, q)), f"t={t}"


class TestSignVerify:
    def test_verify_true(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        for t, sk_t in enumerate(sks):
            assert sel.verify(pk, M, sel.sign(pk, sk_t, M, t), t), f"t={t}"

    def test_wrong_message_rejected(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        sig = sel.sign(pk, sks[0], M, 0)
        assert not sel.verify(pk, b"wrong", sig, 0)

    def test_wrong_epoch_rejected(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        sig = sel.sign(pk, sks[0], M, 0)
        assert not sel.verify(pk, M, sig, 1)

    def test_sigma_size_constant(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        sizes = {len(sel.sign(pk, sk_t, M, t)) for t, sk_t in enumerate(sks)}
        assert len(sizes) == 1

    def test_sk_size_constant(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        sizes = {sk_t.R.nbytes + sk_t.H_tag.nbytes + sk_t.R_trap.nbytes
                 for sk_t in sks}
        assert len(sizes) == 1


class TestGadgetSampler:
    def test_gperp(self, sel_fixture):
        sel, pk, sks, q, n, k, nk, m = sel_fixture
        np.random.seed(SEED)
        gdt = F.GadgetTrapdoor(n, q)
        gs  = F.GadgetSampler(n, q, k)
        r   = sel.sigma / math.sqrt(2)
        for _ in range(20):
            v  = np.random.randint(0, q, n, dtype=np.int64)
            x  = gs.sample_Gperp(v, r)
            Gx = F.mat_mul_mod(gdt.G, x.reshape(-1, 1), q).flatten()
            assert np.all(Gx == v % q)


class TestAdaptiveFSS:
    def test_verify_true(self):
        np.random.seed(SEED)
        q, n, sigma, ell_phf = 97, 8, 8.0, 32
        k  = math.ceil(math.log2(q))
        nk = n * k
        m  = 2 * n + nk
        adp = F.AdaptiveFSS(n=n, q=q, T=4, ell=4, sigma=sigma)
        adp.phf = F.ProgrammableHashFunction(n, m, nk, q, beta=sigma, ell_phf=ell_phf)
        pk, sk0 = adp.key_gen()
        sk1 = adp.key_update(pk, sk0)
        for t, sk_t in enumerate([sk0, sk1]):
            assert adp.verify(pk, M, adp.sign(pk, sk_t, M, t), t), f"t={t}"

    def test_id_b0_fixed(self):
        np.random.seed(SEED)
        q, n, sigma, ell_phf = 97, 8, 8.0, 32
        k  = math.ceil(math.log2(q))
        nk = n * k
        m  = 2 * n + nk
        adp = F.AdaptiveFSS(n=n, q=q, T=4, ell=4, sigma=sigma)
        adp.phf = F.ProgrammableHashFunction(n, m, nk, q, beta=sigma, ell_phf=ell_phf)
        pk, _ = adp.key_gen()
        for t in range(len(pk.IDs)):
            assert pk.IDs[t][0] == 1, f"b_{{t,0}} must be 1 at t={t}"
