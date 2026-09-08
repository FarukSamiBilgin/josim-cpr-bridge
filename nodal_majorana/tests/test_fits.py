"""Tests for the fitting layer, on synthetic data with known answers.

The fits decide the headline numbers, so they are checked against series built
from the very laws they are meant to detect: an exponential with a known
``xi_M`` and ``k_F``, and a power law with a known ``p``.  Each must be
recovered, and each must be *preferred* over the other when it is the truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodal_majorana.fits import (  # noqa: E402
    discriminate,
    envelope,
    fit_exponential,
    fit_exponential_envelope,
    fit_power,
    fit_power_envelope,
    window_stability,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def synth_exponential(xi=3.0, kf=2.0, phi=0.4, amp=0.8, lmax=32.0, step=0.5):
    L = np.arange(4.0, lmax + 1e-9, step)
    return L, amp * L**-0.5 * np.exp(-L / xi) * np.abs(np.cos(kf * L + phi))


def synth_power(p=1.5, k=2.0, phi=0.4, amp=0.8, lmax=60.0, step=0.5):
    L = np.arange(4.0, lmax + 1e-9, step)
    return L, amp * L**-p * np.abs(np.cos(k * L + phi))


def test_envelope_picks_peaks() -> None:
    print("\nenvelope extraction")
    L, y = synth_exponential()
    xe, ye = envelope(L, y)
    check("finds several peaks", xe.size >= 8, f"{xe.size} peaks from {L.size} pts")
    # every accepted point must be at least as large as its neighbours
    idx = {float(a): float(b) for a, b in zip(xe, ye)}
    ok = True
    for a, b in idx.items():
        near = y[np.abs(L - a) <= 0.5 + 1e-9]
        ok &= b >= near.max() - 1e-12
    check("every peak dominates its neighbours", ok)
    # The point of the envelope is that |cos| = 1 there, so the peak values must
    # reproduce the bare decay law rather than the oscillating series.  Compare
    # them with the known envelope amp * L^-1/2 exp(-L/xi) of `synth_exponential`.
    expected = 0.8 * xe**-0.5 * np.exp(-xe / 3.0)
    ratio = ye / expected
    check("peak values sit on the bare decay envelope",
          float(ratio.min()) > 0.80,
          f"peak / envelope in [{ratio.min():.3f}, {ratio.max():.3f}]")


def test_recovers_exponential() -> None:
    print("\nexponential law recovered from synthetic data")
    xi, kf = 3.0, 2.0
    L, y = synth_exponential(xi=xi, kf=kf)
    fe = fit_exponential(L, y)
    check("full fit recovers xi_M", abs(fe.params["xi_M"] - xi) / xi < 0.05,
          f"{fe.params['xi_M']:.4f} vs {xi}")
    check("full fit recovers k_F", abs(fe.params["k_F"] - kf) / kf < 0.05,
          f"{fe.params['k_F']:.4f} vs {kf}")
    fv = fit_exponential_envelope(L, y)
    check("envelope fit recovers xi_M",
          abs(fv.params["xi_M"] - xi) / xi < 0.10,
          f"{fv.params['xi_M']:.4f} vs {xi}")
    d = discriminate(L, y)
    check("discriminate says exponential", d["verdict"] == "exponential",
          f"rms exp {d['exponential'].rms_log:.4f} vs power "
          f"{d['power'].rms_log:.4f}")
    check("verdict flagged decisive", d["decisive"],
          f"L x{d['L_ratio']:.1f}, {d['decades']:.1f} decades")


def test_recovers_power_law() -> None:
    print("\npower law recovered from synthetic data")
    p = 1.5
    L, y = synth_power(p=p)
    fp = fit_power(L, y, with_oscillation=True)
    check("full fit recovers p", abs(fp.params["p"] - p) / p < 0.05,
          f"{fp.params['p']:.4f} vs {p}")
    fv = fit_power_envelope(L, y)
    check("envelope fit recovers p", abs(fv.params["p"] - p) / p < 0.05,
          f"{fv.params['p']:.4f} vs {p}")
    d = discriminate(L, y)
    check("discriminate says power law", d["verdict"] == "power law",
          f"rms exp {d['exponential'].rms_log:.4f} vs power "
          f"{d['power'].rms_log:.4f}")
    pm, ps, _ = window_stability(L, y)
    check("sliding windows agree on p", abs(pm - p) < 0.1 and ps < 0.15,
          f"p = {pm:.3f} +/- {ps:.3f}")


def test_narrow_range_is_not_decisive() -> None:
    print("\na narrow range in L is reported as indecisive")
    L, y = synth_exponential(lmax=11.0)
    d = discriminate(L, y)
    check("short scan is flagged not decisive", not d["decisive"],
          f"L x{d['L_ratio']:.2f} over {d['decades']:.2f} decades")


def main() -> int:
    print("=" * 68)
    print("nodal_majorana / fit tests")
    print("=" * 68)
    test_envelope_picks_peaks()
    test_recovers_exponential()
    test_recovers_power_law()
    test_narrow_range_is_not_decisive()
    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("all fit tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
