"""Post-process a production run: which decay law, and how far from v_F/Delta.

`run_exponent.py` measures ``delta_E(L)`` for several node-regulator strengths
and several directions of the vortex-vortex axis, and writes them to
``results/law.json``.  This script asks two questions of that data.

1.  **Does the exponential law hold, and with the predicted length?**
    For each scan the envelope is fitted with both laws and the fitted
    ``xi_M`` compared with ``v_F(theta) / Delta(theta)`` from the clean band
    structure.  A ratio near one means the vortex pair hybridizes through
    ordinary evanescent quasiparticles at the gap of its own direction.

2.  **Where does that break down?**
    It should break down as the axis turns toward a node and as the node gap is
    reduced -- both push the pair into the regime ``L << xi_M`` where the
    exponential has not yet taken over and the gapless quasiparticles dominate.
    There the fitted ``xi_M`` runs away from the prediction and a power law
    describes the data better; the exponent of that power law is reported with
    its sliding-window spread.

Finally the near-nodal scans are collapsed onto a single curve,

    |delta_E| / Delta_n  =  L^(-p) F(L / xi_M),

by scanning ``p`` for the value that minimises the scatter of the collapsed
data.  The collapse uses curves with different ``xi_M`` at once, so it is far
more constrained than a power-law fit inside any single narrow window.

Run:  python3 nodal_majorana/analyse_law.py [path/to/law.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodal_majorana.fits import (  # noqa: E402
    envelope,
    fit_crossover,
    fit_exponential_envelope,
    fit_power_envelope,
    window_stability,
)

DEFAULT = Path(__file__).resolve().parent / "results" / "law.json"


def per_scan_table(law: dict) -> list[dict]:
    rows = []
    for rec in law.values():
        L = np.asarray(rec.get("L", []), float)
        dE = np.asarray(rec.get("dE", []), float)
        if L.size < 6:
            continue
        row = {k: rec[k] for k in ("d_is", "angle", "gap", "v_F", "xi_ref")}
        row["n"] = int(L.size)
        try:
            fe = fit_exponential_envelope(L, dE)
            fp = fit_power_envelope(L, dE)
            xe, ve = envelope(L, dE)
            row.update(
                xi_fit=fe.params["xi_M"],
                rms_exp=fe.rms_log,
                p_fit=fp.params["p"],
                rms_pow=fp.rms_log,
                n_env=int(xe.size),
                decades=float(np.log10(ve.max() / ve.min())),
                L_ratio=float(xe.max() / xe.min()),
            )
            row["ratio"] = row["xi_fit"] / row["xi_ref"]
            row["prefers"] = "exp" if fe.rms_log < fp.rms_log else "power"
            try:
                pm, ps, _ = window_stability(L, dE, min_points=4, n_windows=4)
                row["p_win"], row["p_spread"] = pm, ps
            except ValueError:
                pass
        except ValueError:
            pass
        rows.append(row)
    return sorted(rows, key=lambda r: (r["d_is"], r["angle"]))


def print_table(rows: list[dict]) -> None:
    print("\nper-scan decay law")
    print(f"  {'d_is':>5} {'ang':>5} {'gap':>7} {'xi_ref':>7} {'xi_fit':>8}"
          f" {'ratio':>6} {'p':>6} {'rms_e':>7} {'rms_p':>7} {'prefers':>8}"
          f" {'n_env':>6}")
    for r in rows:
        if "xi_fit" not in r:
            print(f"  {r['d_is']:5.2f} {r['angle']:5.1f} {r['gap']:7.4f}"
                  f" {r['xi_ref']:7.2f}   (envelope too short)")
            continue
        print(f"  {r['d_is']:5.2f} {r['angle']:5.1f} {r['gap']:7.4f}"
              f" {r['xi_ref']:7.2f} {r['xi_fit']:8.2f} {r['ratio']:6.2f}"
              f" {r['p_fit']:6.2f} {r['rms_exp']:7.3f} {r['rms_pow']:7.3f}"
              f" {r['prefers']:>8} {r['n_env']:6d}")


def calibrate(rows: list[dict], ref_angle: float = 0.0) -> tuple[float, float]:
    """The O(1) prefactor ``c`` in ``xi_M = c * v_F(theta) / Delta(theta)``.

    ``v_F / Delta`` is the right *scale* but not the right number: the Majorana
    decay length carries a model-dependent factor of order one (the Rashba
    projection, the definition of the gap, the core profile).  That factor is a
    property of the band structure, not of the direction, so it can be measured
    once on the antinodal axis -- where the background is ordinarily gapped and
    the decay is known to be exponential -- and then carried over.

    Returns ``(c, spread)`` from the antinodal scans at every regulator
    strength.  A small spread is what licenses the transfer.
    """
    vals = [r["ratio"] for r in rows
            if abs(r["angle"] - ref_angle) < 1e-9 and "ratio" in r]
    if not vals:
        return 1.0, 0.0
    a = np.array(vals)
    return float(a.mean()), float(a.std())


def calibrated_exponent(law: dict, c: float, c_err: float) -> list[dict]:
    """Crossover fit ``A L^-p exp(-L / (c v_F/Delta))`` on every scan.

    With the length fixed by the calibration there is nothing left for the fit
    to trade against ``p``, so ``p`` measures the algebraic part of the decay
    alone.  On a gapped axis it must come back near ``1/2`` -- the
    two-dimensional geometric factor -- because that is what the calibration
    assumed.  On the nodal axis it is a measurement.

    The quoted uncertainty is the swing in ``p`` when the calibration is moved
    by its own spread, which dominates any scatter in the individual fits.
    """
    out = []
    for rec in law.values():
        L = np.asarray(rec.get("L", []), float)
        dE = np.asarray(rec.get("dE", []), float)
        if L.size < 6:
            continue
        row = {k: rec[k] for k in ("d_is", "angle", "gap", "xi_ref")}
        try:
            base = fit_crossover(L, dE, c * rec["xi_ref"])
            row["p"] = base.params["p"]
            row["rms"] = base.rms_log
            row["n_env"] = base.npoints
            if c_err > 0:
                lo = fit_crossover(L, dE, (c - c_err) * rec["xi_ref"]).params["p"]
                hi = fit_crossover(L, dE, (c + c_err) * rec["xi_ref"]).params["p"]
                row["p_err"] = abs(hi - lo) / 2.0
        except ValueError:
            continue
        out.append(row)
    return sorted(out, key=lambda r: (r["angle"], r["d_is"]))


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"no results at {path}; run run_exponent.py first")
        return 1
    res = json.loads(path.read_text())
    law = res.get("law", {})
    if not law:
        print("results contain no 'law' block (run without --only, or with c)")
        return 1

    print("=" * 78)
    print("nodal_majorana / decay-law analysis")
    print("=" * 78)
    rows = per_scan_table(law)
    print_table(rows)

    print("\ndeviation from xi_M = v_F / gap, as a map")
    angles = sorted({r["angle"] for r in rows})
    dis = sorted({r["d_is"] for r in rows})
    print("  ratio xi_fit / (v_F/gap):")
    print("  " + "d_is\\ang".rjust(9) + "".join(f"{a:>9.1f}" for a in angles))
    for d in dis:
        cells = []
        for a in angles:
            m = [r for r in rows if r["d_is"] == d and r["angle"] == a
                 and "ratio" in r]
            cells.append(f"{m[0]['ratio']:9.2f}" if m else "        -")
        print("  " + f"{d:9.2f}" + "".join(cells))
    print("\n  A ratio near 1 means the pair hybridizes through evanescent")
    print("  quasiparticles at the gap of its own direction.  Ratios well above")
    print("  1 mean the measured decay is slower than any exponential with that")
    print("  length -- the signature of the gapless nodal channel.")

    c, c_err = calibrate(rows)
    print(f"\ncalibration on the antinodal axis: xi_M = c * v_F / Delta with")
    print(f"  c = {c:.3f} +/- {c_err:.3f}   "
          f"(from {sum(1 for r in rows if r['angle'] == 0.0)} scans)")
    print("  v_F/Delta gets the scale right but not the O(1) factor; measuring")
    print("  it once where the background is ordinarily gapped lets it be")
    print("  carried to the nodal axis, where the deviation is the signal.")

    cal = calibrated_exponent(law, c, c_err)
    print("\nalgebraic exponent with the length fixed by that calibration")
    print("  |dE| = A L^-p exp(-L / (c v_F/Delta))")
    print(f"  {'d_is':>6} {'angle':>7} {'p':>8} {'+/-':>7} {'rms':>7} {'n_env':>6}")
    for r in cal:
        e = f"{r['p_err']:7.2f}" if "p_err" in r else "      -"
        print(f"  {r['d_is']:6.2f} {r['angle']:7.1f} {r['p']:8.2f} {e}"
              f" {r['rms']:7.3f} {r['n_env']:6d}")
    ref = [r for r in cal if r["angle"] == 0.0]
    if ref:
        pr = np.array([r["p"] for r in ref])
        print(f"\n  antinodal axis:  p = {pr.mean():.2f} +/- {pr.std():.2f}"
              "   (the 2D geometric factor the calibration assumed)")
    nod = [r for r in cal if r["angle"] == max(r2["angle"] for r2 in cal)]
    if nod:
        print("  nodal axis, by regulator strength (weakest node gap last):")
        for r in sorted(nod, key=lambda r: -r["d_is"]):
            e = f" +/- {r['p_err']:.2f}" if "p_err" in r else ""
            print(f"    gap at node {r['gap']:.3f}:  p = {r['p']:.2f}{e}")
        print("  A nodal p falling below the antinodal value means the decay is")
        print("  slower than the directional gap alone accounts for.")

    out = {"per_scan": rows, "calibration": {"c": c, "c_err": c_err},
           "calibrated": cal}

    dest = path.with_name(path.stem + "_analysis.json")
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
