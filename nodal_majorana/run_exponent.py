"""The vortex-vortex Majorana hybridization law of a nodal 2D superconductor.

The question
------------
On a fully gapped topological superconductor the splitting between two
vortex-bound Majorana modes falls exponentially with separation, with a length
``xi_M = v_F / Delta``.  On a *nodal* background the mediating quasiparticles
are gapless along the nodal directions, so the exponential should give way to a
power law ``|dE| ~ L^-p`` -- and the exponent ``p``, together with how the decay
depends on the angle between the vortex axis and the node, is what this run set
out to measure.

Why it has to be regulated
--------------------------
On the bare nodal background a vortex binds no zero mode: the would-be Majorana
is degenerate with the nodal continuum and survives only as a resonance.  Its
lowest core-weighted level does not converge as the box grows (step [B] shows
this directly), so a splitting read off from it would measure the box.

The nodes are therefore regulated by an on-site ``i * d_is`` component -- the
``d + is`` state -- which opens a gap ``~0.69 d_is`` exactly at the nodes,
leaves the antinodal gap essentially untouched, and makes the Chern number odd,
so a genuine bound Majorana exists and ``delta_E(L)`` is unambiguous.  The nodal
hybridization length then scales as ``1 / d_is``, and the separations that fit
in an affordable box straddle it:

    L >> xi_M     exponential decay at the gap of the vortex axis
    L <~ xi_M     the gapless channel still dominates -- the power-law regime

Sending ``d_is -> 0`` widens the second window but also closes the
Caroli-de Gennes-Matricon minigap, which tracks the node gap (``E_1 ~ 0.24
Delta_n``, measured in step [A]).  Since the pair is a resolvable two-level
system only while ``|dE| < E_1``, the two requirements pull against each other:
at the weakest regulator every separation that fails the filters lies in the
sub-``xi_M`` window.  Step [D] measures where the usable window sits.

What the run produces
---------------------
[A] the nodal structure, and what the regulator does to it
[B] the absence of a bound zero mode without the regulator
[C] delta_E(L) for each regulator strength and axis direction, every point
    converged in box size, each scan fitted on its envelope and compared with
    v_F(theta)/Delta(theta) from the clean band structure
[D] the interval of separations over which the pair exists as a Majorana pair

Use ``--deep`` for weaker regulators and a wider L range on the nodal and
antinodal axes only.  That range matters: measured over L in [4, 20] the nodal
scan at d_is = 0.5 gives xi_fit/xi_ref = 3.75 and looks non-exponential, while
the same scan over L in [3, 32] gives 1.23.  A narrow window in L cannot tell
the two laws apart, so ``--deep`` is the run whose numbers should be quoted.

`analyse_law.py` reads the resulting JSON and extracts the decay law, the O(1)
calibration of xi_M = c v_F/Delta, and the algebraic exponent.

Run:  python3 nodal_majorana/run_exponent.py [--quick | --deep]
      python3 nodal_majorana/analyse_law.py [results/law_deep.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodal_majorana.fits import (  # noqa: E402
    discriminate,
    envelope,
    fit_exponential_envelope,
)
from nodal_majorana.model import Model  # noqa: E402
from nodal_majorana.solve import (  # noqa: E402
    converged_splitting,
    single_vortex_energy,
)

# Normal state shared by every model below; only the pairing ever changes.
NORMAL = dict(mu=0.4, vz=2.4, alpha=1.6, d0=1.6, xi0=1.5)
GAPPED = Model(form="s", **NORMAL)        # gapped benchmark, Chern -1
BARE_NODAL = Model(form="d", **NORMAL)    # nodes on the zone diagonals

# Regulator strengths at which the Majorana pair is a resolvable two-level
# system (see the module docstring); these span a factor 2.8 in the nodal gap.
REGULATORS = (0.5, 0.7, 1.0, 1.4)
ANGLES = (0.0, 22.5, 45.0)
SEPS = np.arange(4.0, 20.1, 0.8)
NODAL_ANGLE = 45.0
TAG = "law"

OUT = Path(__file__).resolve().parent / "results"


def gap_along(model: Model, angle_deg: float, n: int = 1501) -> float:
    """Smallest BdG eigenvalue along a ray from Gamma."""
    ang = np.deg2rad(angle_deg)
    return min(
        float(np.min(np.abs(np.linalg.eigvalsh(
            model.bdg_k(k * np.cos(ang), k * np.sin(ang))))))
        for k in np.linspace(0.0, np.pi, n)
    )


def reference_xi(model: Model, angle_deg: float) -> tuple[float, float, float]:
    """``(k_F, v_F, xi_M = v_F / gap)`` along a ray, from the clean bands."""
    kf, vf = model.fermi_surface(angle_deg)[0]
    return kf, vf, vf / max(gap_along(model, angle_deg), 1e-12)


def step_a() -> dict:
    print("\n[A] the background is nodal, and the regulator gaps the nodes")
    print(f"      {'angle':>7} {'gap (d-wave)':>14} {'gap (s-wave)':>14}")
    angles = [0, 15, 30, 40, 45, 50, 60, 75, 90]
    gd = [gap_along(BARE_NODAL, a) for a in angles]
    for a, g in zip(angles, gd):
        mark = "   <- node" if g < 5e-3 else ""
        print(f"      {a:7g} {g:14.6f} {gap_along(GAPPED, a):14.6f}{mark}")

    print(f"\n      {'d_is':>6} {'Chern':>6} {'gap@45':>10} {'gap@0':>10}"
          f" {'CdGM E_1':>10} {'E_1/gap@45':>11}")
    reg = []
    for dis in (0.0,) + REGULATORS:
        m = Model(form="d", d_is=dis, **NORMAL)
        c = m.chern_number(n=40) if dis > 0 else None
        gn, ga = gap_along(m, NODAL_ANGLE), gap_along(m, 0.0)
        _, _, e1 = single_vortex_energy(m, 64, 64)
        reg.append({"d_is": dis, "chern": c, "gap_node": gn,
                    "gap_antinode": ga, "minigap": e1})
        print(f"      {dis:6.2f} {str(c):>6} {gn:10.5f} {ga:10.5f}"
              f" {e1:10.5f} {e1 / max(gn, 1e-9):11.3f}")
    print("      The node gap is linear in d_is while the antinodal gap barely")
    print("      moves, so the regulator acts only where the nodes are.  The")
    print("      CdGM minigap tracks the *node* gap, not the maximum gap --")
    print("      which is what closes the power-law window.")
    return {"gap_vs_angle": {"angles": angles, "gap_d": gd}, "regulator": reg}


def step_b() -> dict:
    print("\n[B] on the bare nodal background a vortex binds no zero mode")
    print(f"      {'box':>5} {'|E| bare d':>13} {'w':>6} | {'|E| d+is=1.0':>13}"
          f" {'w':>6} | {'|E| s-wave':>13} {'w':>6}")
    reg = Model(form="d", d_is=1.0, **NORMAL)
    rows = []
    for L in (32, 40, 48, 56, 64, 72):
        e0, w0, _ = single_vortex_energy(BARE_NODAL, L, L)
        e1, w1, _ = single_vortex_energy(reg, L, L)
        e2, w2, _ = single_vortex_energy(GAPPED, L, L)
        rows.append({"box": L, "bare": e0, "reg": e1, "gapped": e2,
                     "w_bare": w0, "w_reg": w1, "w_gapped": w2})
        print(f"      {L:5d} {e0:13.5e} {w0:6.3f} | {e1:13.5e} {w1:6.3f} |"
              f" {e2:13.5e} {w2:6.3f}")
    fb = rows[0]["bare"] / rows[-1]["bare"]
    fg = rows[0]["gapped"] / rows[-1]["gapped"]
    print(f"      across these boxes the bare-nodal level falls by x{fb:.1f}"
          f" (core weight ~{rows[-1]['w_bare']:.2f}),")
    print(f"      the gapped Majorana by x{fg:.3g} (core weight"
          f" ~{rows[-1]['w_gapped']:.2f}).")
    print("      The bare column is a finite-size continuum level, not a bound")
    print("      state: its 'splitting' would measure the box, not the physics.")
    return {"single_vortex": rows, "factor_bare": fb, "factor_gapped": fg}


def scan(model: Model, angle: float, seps, pads, label: str):
    xs, ys, bad = [], [], []
    t0 = time.time()
    print(f"\n      {label} -- axis at {angle:g} deg")
    print(f"      {'L':>6} {'|dE|':>13} {'box':>5} {'w':>6} {'E_1':>9}  status")
    for sep in seps:
        r, hist = converged_splitting(model, float(sep), angle, pads=pads)
        if r is None:
            last = hist[-1]
            bad.append(float(sep))
            print(f"      {sep:6.2f} {abs(last.delta_e):13.5e} {last.lx:5d} "
                  f"{last.w_selected:6.3f} {last.minigap:9.4f}  dropped")
            continue
        xs.append(float(sep))
        ys.append(abs(r.delta_e))
        print(f"      {sep:6.2f} {abs(r.delta_e):13.5e} {r.lx:5d} "
              f"{r.w_selected:6.3f} {r.minigap:9.4f}  ok")
    print(f"      ({time.time() - t0:.0f}s, {len(xs)}/{len(seps)} converged,"
          f" {len(bad)} dropped)")
    return np.array(xs), np.array(ys), bad


def step_c(quick: bool) -> dict:
    print("\n[C] the hybridization law, direction by direction")
    print("      For each regulator strength and each axis, delta_E(L) is fitted")
    print("      on its envelope and the resulting xi_M compared with v_F/gap")
    print("      taken from the clean band structure along the same ray.  The")
    print("      fit is never shown the reference value.")
    # Step 0.8 keeps about 3.6 samples per oscillation period (2 pi / k_F
    # ~ 2.9 a), which is what the neighbour-based envelope needs to find
    # peaks rather than alias them.
    seps = np.arange(4.0, 16.1, 1.2) if quick else SEPS
    pads = (12, 20) if quick else (12, 20, 28)

    out = {}
    for dis in REGULATORS:
        m = Model(form="d", d_is=dis, **NORMAL)
        for ang in ANGLES:
            kf, vf, xi_ref = reference_xi(m, ang)
            gap = gap_along(m, ang)
            print(f"\n      d_is={dis}, {ang:g} deg:  k_F={kf:.3f}, v_F={vf:.3f},"
                  f" gap={gap:.4f}  ->  xi_M(ref) = {xi_ref:.2f} a")
            x, y, bad = scan(m, ang, seps, pads, f"d+is {dis}")
            rec = {"d_is": dis, "angle": ang, "k_F": kf, "v_F": vf,
                   "gap": gap, "xi_ref": xi_ref, "dropped": bad,
                   "L": x.tolist(), "dE": y.tolist()}
            if x.size >= 6:
                try:
                    fe = fit_exponential_envelope(x, y)
                    d = discriminate(x, y)
                    xe, _ = envelope(x, y)
                    rec.update(xi_fit=fe.params["xi_M"], rms=fe.rms_log,
                               n_env=int(xe.size), verdict=d["verdict"],
                               decades=d["decades"])
                    err = abs(fe.params["xi_M"] - xi_ref) / xi_ref
                    print(f"        {fe}")
                    print(f"        xi_M fit / ref = "
                          f"{fe.params['xi_M'] / xi_ref:.3f}  ({100 * err:.1f}%)"
                          f"   envelope prefers: {d['verdict']}")
                except ValueError as exc:
                    print(f"        envelope fit unavailable: {exc}")
            out[f"dis{dis}_ang{ang:g}"] = rec
    return {"law": out}


def step_d() -> dict:
    """Map the window in which the pair exists as a resolvable two-level system.

    Two conditions have to hold at once for ``delta_E(L)`` to mean anything:
    the two levels must sit inside the CdGM minigap (otherwise they are core
    ladder states, not a Majorana pair), and the measurement must be converged
    in box size.  This step reports the resulting interval in units of
    ``xi_M``, which is what decides whether the nodal power-law regime
    ``L << xi_M`` is observable at all.
    """
    print("\n[D] the window in which a Majorana pair exists as such")
    print("      |dE| < E_1 (inside the core-level minigap) AND box-converged.")
    print(f"\n      {'d_is':>6} {'xi_M(45)':>9} {'E_1':>8} {'L range':>13}"
          f" {'in units of xi_M':>18}")
    rows = []
    for dis in REGULATORS:
        m = Model(form="d", d_is=dis, **NORMAL)
        _, _, xi_ref = reference_xi(m, NODAL_ANGLE)
        ok, e1 = [], float("nan")
        for sep in np.arange(2.0, 26.1, 1.0):
            r, _ = converged_splitting(m, float(sep), NODAL_ANGLE,
                                       pads=(12, 20, 28))
            if r is None:
                continue
            e1 = r.minigap
            if abs(r.delta_e) < r.minigap:
                ok.append(float(sep))
        if ok:
            lo, hi = min(ok), max(ok)
            rows.append({"d_is": dis, "xi_M": xi_ref, "minigap": e1,
                         "L_min": lo, "L_max": hi,
                         "u_min": lo / xi_ref, "u_max": hi / xi_ref,
                         "n_ok": len(ok)})
            print(f"      {dis:6.2f} {xi_ref:9.2f} {e1:8.4f}"
                  f" {lo:5.1f} - {hi:5.1f} {lo / xi_ref:8.2f} -"
                  f" {hi / xi_ref:6.2f}")
        else:
            rows.append({"d_is": dis, "xi_M": xi_ref, "minigap": e1,
                         "L_min": None, "L_max": None, "n_ok": 0})
            print(f"      {dis:6.2f} {xi_ref:9.2f} {e1:8.4f}"
                  f"   (no resolvable separation)")
    us = [r["u_min"] for r in rows if r.get("u_min") is not None]
    if us:
        print(f"\n      The window opens at L ~ {np.mean(us):.2f} xi_M.")
        print("      Below that the two levels are no longer separated from the")
        print("      core ladder, so the sub-xi_M power-law regime is only")
        print("      partially observable -- and it shrinks as the nodes are")
        print("      approached, because the minigap closes with the node gap.")
    return {"window": rows}


def summarise(res: dict) -> None:
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    rows = [r for r in res.get("law", {}).values() if "xi_fit" in r]
    if rows:
        print("\n  measured hybridization length vs v_F/gap along the same ray:")
        print(f"    {'d_is':>6} {'angle':>7} {'gap':>8} {'xi_ref':>8}"
              f" {'xi_fit':>8} {'ratio':>7} {'law':>12}")
        for r in sorted(rows, key=lambda r: (r["d_is"], r["angle"])):
            print(f"    {r['d_is']:6.2f} {r['angle']:7.1f} {r['gap']:8.4f}"
                  f" {r['xi_ref']:8.2f} {r['xi_fit']:8.2f}"
                  f" {r['xi_fit'] / r['xi_ref']:7.3f} {r.get('verdict', '?'):>12}")
        ratios = np.array([r["xi_fit"] / r["xi_ref"] for r in rows])
        print(f"\n    xi_fit / (v_F/gap) = {ratios.mean():.3f} +/- "
              f"{ratios.std():.3f}  over {ratios.size} independent scans")
        nod = [r for r in rows if r["angle"] == NODAL_ANGLE]
        anti = [r for r in rows if r["angle"] == 0.0]
        pairs = [(a, n) for a in anti for n in nod if n["d_is"] == a["d_is"]]
        if pairs:
            # The calibration factor cancels from a ratio taken within one
            # model, so this is the sharper comparison of the two.
            print("\n  anisotropy (nodal axis vs antinodal axis), same model:")
            for a, n in pairs:
                print(f"    d_is={a['d_is']:<5} xi_M(45)/xi_M(0) = "
                      f"{n['xi_fit'] / a['xi_fit']:.2f}   predicted "
                      f"{n['xi_ref'] / a['xi_ref']:.2f}")
    win = res.get("window")
    if win:
        print("\n  separations over which the pair exists as a Majorana pair:")
        for w in win:
            if w.get("L_min") is None:
                print(f"    d_is={w['d_is']:<5} none resolvable")
                continue
            print(f"    d_is={w['d_is']:<5} L = {w['L_min']:.0f}-{w['L_max']:.0f} a"
                  f"  =  {w['u_min']:.2f}-{w['u_max']:.2f} xi_M"
                  f"   ({w['n_ok']} separations)")
        us = [w["u_min"] for w in win if w.get("u_min") is not None]
        if us:
            print(f"    the window reaches down to L ~ {min(us):.2f} xi_M, so the")
            print("    sub-xi_M regime is partly accessible; run analyse_law.py")
            print("    to extract the decay law from it.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="abcd")
    ap.add_argument("--deep", action="store_true",
                    help="weaker regulators and a wider L range, on the nodal "
                         "and antinodal axes only; writes law_deep.json")
    args = ap.parse_args()

    global REGULATORS, ANGLES, SEPS, TAG
    if args.deep:
        # Weaker regulators push xi_M up and open the sub-xi_M window; the
        # antinodal axis is kept at the same d_is values because it calibrates
        # the O(1) prefactor in xi_M = c * v_F / Delta.
        REGULATORS = (0.4, 0.5, 0.7, 1.0)
        ANGLES = (0.0, 45.0)
        SEPS = np.arange(3.0, 32.1, 0.7)
        TAG = "law_deep"

    print("=" * 76)
    print("vortex-vortex Majorana hybridization on a nodal superconductor")
    print("=" * 76)
    print(f"normal state: {NORMAL}")
    print(f"gapped reference: form='s', Chern = {GAPPED.chern_number(n=48)}")

    res: dict = {"normal": NORMAL, "regulators": list(REGULATORS)}
    if "a" in args.only:
        res.update(step_a())
    if "b" in args.only:
        res.update(step_b())
    if "c" in args.only:
        res.update(step_c(args.quick))
    if "d" in args.only:
        res.update(step_d())
    # Persist before summarising: the summary is presentation only, and a
    # formatting slip there must never cost a completed run its data.
    OUT.mkdir(exist_ok=True)
    path = OUT / (f"{TAG}_quick.json" if args.quick else f"{TAG}.json")
    path.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {path}")

    summarise(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
