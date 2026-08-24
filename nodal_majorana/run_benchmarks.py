"""Benchmark the two-vortex machinery in limits where the answer is known.

Nothing here is a new result.  These are the checks that have to pass before the
nodal number in ``run_exponent.py`` means anything:

  1.  The gapped s-wave background is a Chern-odd topological superconductor, so
      a single vortex binds one Majorana mode.  Its energy under open boundaries
      is the vortex-edge hybridization and must fall exponentially with the box.
  2.  Two vortices in that background must reproduce the textbook law
      ``|dE| ~ L^-1/2 exp(-L/xi_M) |cos(k_F L + phi)|``, with ``xi_M`` and
      ``k_F`` agreeing with values computed independently from the clean band
      structure -- ``xi_M = v_F / Delta_gap`` and ``k_F`` from the Fermi surface.
      The fit is never told either number.
  3.  The same data must be fitted *badly* by a power law, so that the fit which
      later selects a power law in the nodal case is doing real work.

Run:  python3 nodal_majorana/run_benchmarks.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodal_majorana.fits import discriminate, fit_exponential, fit_power  # noqa: E402
from nodal_majorana.model import Model  # noqa: E402
from nodal_majorana.solve import (  # noqa: E402
    converged_splitting,
    single_vortex_energy,
)

# Gapped topological reference point: Chern number -1, bulk gap ~0.75 t.
# The parameters are chosen for a *short* Majorana length (xi_M ~ 2.8 a) so
# that the accessible separations span several decades of splitting -- over a
# narrow range an exponential and a power law are not distinguishable.
GAPPED = Model(mu=0.4, vz=2.4, d0=1.6, alpha=1.6, form="s", xi0=1.5)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def benchmark_1_zero_mode() -> None:
    print("\n[1] a single vortex binds a Majorana mode")
    c = GAPPED.chern_number(n=48)
    gap = GAPPED.bulk_gap(n=161)
    print(f"      Chern number C = {c}, bulk gap = {gap:.4f} t")
    check("Chern number is odd", c % 2 != 0, f"C = {c}")

    boxes = (24, 32, 40, 48, 56, 64)
    es = []
    for L in boxes:
        e, w, minigap = single_vortex_energy(GAPPED, L, L)
        es.append(e)
        print(f"      box {L:3d}:  |E| = {e:.4e}   w_core = {w:.3f}   "
              f"minigap = {minigap:.4f}")
    check("core state sits far below the CdGM minigap", es[-1] < 0.02 * minigap,
          f"{es[-1]:.2e} vs {minigap:.4f}")
    check("vortex-edge splitting falls with box size",
          es[-1] < 0.05 * es[0], f"{es[0]:.2e} -> {es[-1]:.2e}")


def scan(model: Model, angle: float, seps, pads, label: str):
    """Converged delta-E(L) scan; returns (L, dE) for the converged points."""
    xs, ys = [], []
    t0 = time.time()
    print(f"\n      {label}: separation scan at {angle:g} deg")
    print(f"      {'L':>6} {'|dE|':>13} {'box':>5} {'w':>6} {'minigap':>9}  status")
    for sep in seps:
        r, hist = converged_splitting(model, float(sep), angle, pads=pads)
        if r is None:
            last = hist[-1]
            print(f"      {sep:6.1f} {abs(last.delta_e):13.5e} {last.lx:5d} "
                  f"{last.w_selected:6.3f} {last.minigap:9.4f}  NOT CONVERGED")
            continue
        xs.append(float(sep))
        ys.append(abs(r.delta_e))
        print(f"      {sep:6.1f} {abs(r.delta_e):13.5e} {r.lx:5d} "
              f"{r.w_selected:6.3f} {r.minigap:9.4f}  ok")
    print(f"      ({time.time() - t0:.0f}s, {len(xs)}/{len(seps)} converged)")
    return np.array(xs), np.array(ys)


def benchmark_2_exponential_law() -> None:
    print("\n[2] two vortices reproduce the gapped hybridization law")

    fs = GAPPED.fermi_surface(0.0)
    kf_ref, vf_ref = fs[0]
    gap = GAPPED.bulk_gap(n=161)
    xi_ref = vf_ref / gap
    print(f"      independent references from the clean band structure:")
    print(f"        k_F  = {kf_ref:.4f} 1/a      (Fermi surface along 0 deg)")
    print(f"        v_F  = {vf_ref:.4f} t*a")
    print(f"        xi_M = v_F / gap = {xi_ref:.3f} a")

    seps = np.arange(5.0, 32.5, 0.5)
    x, y = scan(GAPPED, 0.0, seps, pads=(12, 20, 28, 36), label="gapped s-wave")

    fe = fit_exponential(x, y)
    fp = fit_power(x, y, with_oscillation=True)
    print("\n      fits (never shown the reference values):")
    print(f"        {fe}")
    print(f"        {fp}")

    xi_fit, kf_fit = fe.params["xi_M"], fe.params["k_F"]
    d_xi = abs(xi_fit - xi_ref) / xi_ref
    d_kf = abs(kf_fit - kf_ref) / kf_ref
    check("fitted xi_M matches v_F / gap within 25%", d_xi < 0.25,
          f"fit {xi_fit:.3f} vs ref {xi_ref:.3f}  ({100 * d_xi:.1f}%)")
    check("fitted k_F matches the Fermi surface within 10%", d_kf < 0.10,
          f"fit {kf_fit:.4f} vs ref {kf_ref:.4f}  ({100 * d_kf:.1f}%)")
    d = discriminate(x, y)
    print("\n      envelope discrimination (oscillation factored out):")
    print(f"        {d['exponential']}")
    print(f"        {d['power']}")
    print(f"        span: {d['decades']:.2f} decades over L x{d['L_ratio']:.1f}"
          f"   verdict: {d['verdict']}  decisive={d['decisive']}")
    check("envelope prefers the exponential law", d["verdict"] == "exponential",
          f"rms exp {d['exponential'].rms_log:.4f} vs power "
          f"{d['power'].rms_log:.4f}")
    check("the L range is wide enough for that verdict to mean something",
          d["decisive"], f"L range x{d['L_ratio']:.1f}, {d['decades']:.2f} decades")
    xi_env = d["exponential"].params["xi_M"]
    check("envelope xi_M also matches v_F / gap within 25%",
          abs(xi_env - xi_ref) / xi_ref < 0.25,
          f"env {xi_env:.3f} vs ref {xi_ref:.3f}")


def main() -> int:
    print("=" * 74)
    print("nodal_majorana / benchmarks in known limits")
    print("=" * 74)
    print(f"model: {GAPPED}")
    benchmark_1_zero_mode()
    benchmark_2_exponential_law()
    print("\n" + "=" * 74)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("all benchmarks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
