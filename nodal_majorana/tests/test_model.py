"""Correctness tests for the real-space BdG builder.

The load-bearing test is `test_realspace_matches_bloch`: it assembles the
Hamiltonian in real space with periodic boundaries, Fourier transforms it back,
and demands agreement with the independently written 4x4 Bloch matrix at every
allowed momentum.  Every sign, factor of two and 1/(2i) in the real-space
correspondence is pinned by that comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodal_majorana.model import (  # noqa: E402
    Lattice,
    Model,
    VortexConfig,
    build_hamiltonian,
    two_vortices,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def bloch_from_realspace(h, lx: int, ly: int, kx: float, ky: float) -> np.ndarray:
    """Project a periodic real-space H onto the plane wave at momentum (kx, ky).

    With the Nambu spinor Psi_k = N^{-1/2} sum_r e^{-i k.r} Psi_r, every one of
    the four components carries the same phase, so a single plane wave suffices.
    """
    lat = Lattice(lx, ly)
    phase = np.exp(1j * (kx * lat.xs + ky * lat.ys)) / np.sqrt(lat.n)
    v = np.zeros((4 * lat.n, 4), dtype=complex)
    for a in range(4):
        v[4 * np.arange(lat.n) + a, a] = phase
    return v.conj().T @ (h @ v)


def test_realspace_matches_bloch() -> None:
    print("\nreal-space assembly vs Bloch Hamiltonian")
    lx, ly = 6, 6
    cases = (
        ("s", 1.0, 0.0),
        ("ext-s", 0.3, 0.0),
        ("d", 1.0, 0.0),
        ("d", 1.0, 0.35),      # d + is: the node regulator
        ("s", 1.0, 0.2),
    )
    for form, ds, d_is in cases:
        m = Model(t=1.0, mu=0.8, alpha=0.7, vz=1.3, d0=0.45, form=form, ds=ds,
                  d_is=d_is)
        h = build_hamiltonian(m, lx, ly, vortices=None, periodic=True)
        worst = 0.0
        for nx in range(lx):
            for ny in range(ly):
                kx, ky = 2 * np.pi * nx / lx, 2 * np.pi * ny / ly
                got = bloch_from_realspace(h, lx, ly, kx, ky)
                want = m.bdg_k(kx, ky)
                worst = max(worst, float(np.max(np.abs(got - want))))
        check(f"form={form!r} d_is={d_is} reproduces bdg_k", worst < 1e-11,
              f"max dev {worst:.2e}")


def test_hermitian() -> None:
    print("\nhermiticity (open boundaries, two vortices)")
    for form, d_is in (("s", 0.0), ("ext-s", 0.0), ("d", 0.0), ("d", 0.35)):
        m = Model(form=form, ds=0.3, d_is=d_is)
        h = build_hamiltonian(m, 14, 14, two_vortices(14, 14, 6.0, 30.0))
        dev = float(abs(h - h.getH()).max())
        check(f"form={form!r} d_is={d_is} H = H^dag", dev < 1e-12,
              f"max dev {dev:.2e}")


def test_particle_hole_symmetry() -> None:
    print("\nparticle-hole symmetry of the spectrum")
    for form, d_is in (("s", 0.0), ("ext-s", 0.0), ("d", 0.0), ("d", 0.35)):
        m = Model(form=form, ds=0.3, d_is=d_is)
        h = build_hamiltonian(m, 12, 12, two_vortices(12, 12, 5.0, 45.0))
        ev = np.linalg.eigvalsh(h.toarray())
        dev = float(np.max(np.abs(ev + ev[::-1])))
        check(f"form={form!r} d_is={d_is} spectrum symmetric about 0", dev < 1e-9,
              f"max |E_i + E_-i| = {dev:.2e}")


def test_vortex_profile() -> None:
    print("\nvortex order-parameter profile")
    cfg = two_vortices(41, 41, 12.0, 0.0)
    (c1x, c1y), (c2x, c2y) = cfg.centers

    # winding of the phase on a small loop about one core must be +1
    th = np.linspace(0, 2 * np.pi, 4001)
    r = 2.0
    z = cfg.order_parameter(c1x + r * np.cos(th), c1y + r * np.sin(th), 2.0)
    dphi = np.diff(np.unwrap(np.angle(z)))
    winding = float(np.sum(dphi) / (2 * np.pi))
    check("winding about one core = +1", abs(winding - 1.0) < 1e-3,
          f"got {winding:.6f}")

    # winding on a large loop enclosing both cores must be +2
    rr = 18.0
    cx, cy = 20.0, 20.0
    z2 = cfg.order_parameter(cx + rr * np.cos(th), cy + rr * np.sin(th), 2.0)
    w2 = float(np.sum(np.diff(np.unwrap(np.angle(z2)))) / (2 * np.pi))
    check("winding about both cores = +2", abs(w2 - 2.0) < 1e-3, f"got {w2:.6f}")

    # amplitude vanishes at the cores and saturates far away
    amp_core = abs(complex(cfg.order_parameter(c1x, c1y, 2.0)))
    amp_far = abs(complex(cfg.order_parameter(c1x + 60.0, c1y, 2.0)))
    check("|Delta| -> 0 at the core", amp_core < 1e-12, f"{amp_core:.2e}")
    check("|Delta| -> 1 far away", abs(amp_far - 1.0) < 1e-6, f"{amp_far:.8f}")


def test_uniform_limit_recovers_no_vortex() -> None:
    print("\nzero-vortex limit")
    m = Model(form="s")
    a = build_hamiltonian(m, 10, 10, VortexConfig())
    b = build_hamiltonian(m, 10, 10, None)
    check("empty VortexConfig == no vortices", float(abs(a - b).max()) < 1e-14)


def test_gap_and_topology() -> None:
    print("\nbulk gap and band-inversion criterion")
    gapped = Model(form="s", mu=1.0, vz=1.2, d0=0.5, alpha=1.0)
    check("s-wave parameters are topological at Gamma",
          gapped.is_topological_at_gamma(),
          f"Vz^2={gapped.vz**2:.3f} > D0^2+mu^2={gapped.d0**2 + gapped.mu**2:.3f}")
    g = gapped.bulk_gap(n=121)
    check("s-wave bulk gap is finite", g > 1e-3, f"gap = {g:.4f} t")

    nodal = Model(form="d", mu=1.0, vz=1.2, d0=0.5, alpha=1.0)
    gn_coarse = nodal.bulk_gap(n=81)
    gn_fine = nodal.bulk_gap(n=321)
    check("d-wave gap collapses under mesh refinement",
          gn_fine < 0.5 * gn_coarse,
          f"{gn_coarse:.5f} -> {gn_fine:.5f}")


def main() -> int:
    print("=" * 68)
    print("nodal_majorana / model tests")
    print("=" * 68)
    test_realspace_matches_bloch()
    test_hermitian()
    test_particle_hole_symmetry()
    test_vortex_profile()
    test_uniform_limit_recovers_no_vortex()
    test_gap_and_topology()
    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("all model tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
