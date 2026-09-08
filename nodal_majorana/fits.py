"""Fitting the measured delta-E(L) to the two competing hybridization laws.

Gapped background
-----------------
Two Majorana modes separated by ``L`` in a fully gapped two-dimensional
superconductor hybridize through evanescent quasiparticles:

    |delta_E(L)| = A * L^(-1/2) * exp(-L / xi_M) * |cos(k_F L + phi)|

-- exponential decay, an oscillation at the Fermi wavelength, and the 1/sqrt(L)
of two-dimensional propagation.

Nodal background
----------------
Along a nodal direction the mediating quasiparticles are gapless, the
exponential factor is absent, and what remains is a power law

    |delta_E(L)| = A * L^(-p) * |cos(k_n L + phi)|

whose exponent ``p`` is the quantity this package exists to measure.

Because both laws carry an oscillation that drives |delta_E| through zero, the
exponent is extracted twice and by different routes: once from a full nonlinear
fit to every point, and once from the envelope alone (the local maxima, where
the oscillation factor is one).  The two must agree, and the quoted uncertainty
comes from refitting over sliding windows in L rather than from a single
covariance matrix -- window drift, not measurement noise, is what limits a
power-law exponent extracted over a finite range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

_FLOOR = 1e-4  # keeps log|cos| finite at the oscillation nodes


@dataclass
class FitResult:
    name: str
    params: dict[str, float]
    rms_log: float
    npoints: int

    def __str__(self) -> str:
        ps = "  ".join(f"{k}={v:.5g}" for k, v in self.params.items())
        return f"{self.name:<12} {ps}   rms(log) = {self.rms_log:.4f}  (n={self.npoints})"


def _rms(res: np.ndarray) -> float:
    return float(np.sqrt(np.mean(res**2)))


def fit_exponential(lengths, values) -> FitResult:
    """Fit ``A L^-1/2 exp(-L/xi) |cos(kF L + phi)|`` in log space."""
    x = np.asarray(lengths, float)
    y = np.log(np.abs(np.asarray(values, float)))

    def resid(par):
        loga, inv_xi, kf, phi = par
        model = (
            loga
            - 0.5 * np.log(x)
            - inv_xi * x
            + np.log(np.maximum(np.abs(np.cos(kf * x + phi)), _FLOOR))
        )
        return y - model

    best = None
    for kf0 in np.linspace(0.2, np.pi, 25):
        for phi0 in np.linspace(0.0, np.pi, 4, endpoint=False):
            try:
                s = least_squares(
                    resid,
                    x0=[y[0] + 0.5 * np.log(x[0]), 0.25, kf0, phi0],
                    loss="soft_l1",
                    f_scale=0.6,
                    max_nfev=4000,
                )
            except Exception:  # pragma: no cover - optimizer edge cases
                continue
            if best is None or s.cost < best.cost:
                best = s
    loga, inv_xi, kf, phi = best.x
    return FitResult(
        "exponential",
        {
            "A": float(np.exp(loga)),
            "xi_M": float(1.0 / inv_xi) if inv_xi != 0 else np.inf,
            "k_F": float(abs(kf)),
            "phi": float(np.mod(phi, 2 * np.pi)),
        },
        _rms(resid(best.x)),
        x.size,
    )


def fit_power(lengths, values, with_oscillation: bool = True) -> FitResult:
    """Fit ``A L^-p |cos(k L + phi)|`` (or the bare power law) in log space."""
    x = np.asarray(lengths, float)
    y = np.log(np.abs(np.asarray(values, float)))

    if not with_oscillation:
        def resid0(par):
            loga, p = par
            return y - (loga - p * np.log(x))

        s = least_squares(resid0, x0=[y[0], 2.0], loss="soft_l1", f_scale=0.6)
        return FitResult(
            "power",
            {"A": float(np.exp(s.x[0])), "p": float(s.x[1])},
            _rms(resid0(s.x)),
            x.size,
        )

    def resid(par):
        loga, p, k, phi = par
        model = (
            loga
            - p * np.log(x)
            + np.log(np.maximum(np.abs(np.cos(k * x + phi)), _FLOOR))
        )
        return y - model

    best = None
    for k0 in np.linspace(0.2, np.pi, 25):
        for phi0 in np.linspace(0.0, np.pi, 4, endpoint=False):
            try:
                s = least_squares(
                    resid,
                    x0=[y[0], 2.0, k0, phi0],
                    loss="soft_l1",
                    f_scale=0.6,
                    max_nfev=4000,
                )
            except Exception:  # pragma: no cover
                continue
            if best is None or s.cost < best.cost:
                best = s
    loga, p, k, phi = best.x
    return FitResult(
        "power+osc",
        {
            "A": float(np.exp(loga)),
            "p": float(p),
            "k": float(abs(k)),
            "phi": float(np.mod(phi, 2 * np.pi)),
        },
        _rms(resid(best.x)),
        x.size,
    )


def envelope(lengths, values, min_spacing: float = 0.0
             ) -> tuple[np.ndarray, np.ndarray]:
    """Peaks of |values|: the turning points of the oscillation.

    At a peak the oscillation factor is one, so a decay law fitted to the
    envelope carries no assumption about the oscillation wavevector.

    Peaks are found against immediate neighbours rather than over a sliding
    window.  A window comparison is wrong here: the signal decays, so a window
    as wide as one oscillation period rejects every peak except the first, and
    a window narrower than a period is just a slower way of testing neighbours.
    The sampling in ``L`` must therefore resolve the oscillation -- at least two
    points per period -- which is the caller's responsibility.

    ``min_spacing`` optionally thins peaks that sit closer than that in ``L``,
    keeping the earlier of the pair.
    """
    x = np.asarray(lengths, float)
    v = np.abs(np.asarray(values, float))
    if x.size < 3:
        return x, v
    keep = [i for i in range(1, x.size - 1) if v[i] >= v[i - 1] and v[i] >= v[i + 1]]
    if v[0] >= v[1]:
        keep.insert(0, 0)
    if v[-1] >= v[-2]:
        keep.append(x.size - 1)
    if min_spacing > 0 and keep:
        thinned = [keep[0]]
        for i in keep[1:]:
            if x[i] - x[thinned[-1]] >= min_spacing:
                thinned.append(i)
        keep = thinned
    idx = np.array(keep, dtype=int)
    return x[idx], v[idx]


def fit_power_envelope(lengths, values, min_spacing: float = 0.0) -> FitResult:
    """Straight-line fit of log(envelope) vs log(L)."""
    xe, ve = envelope(lengths, values, min_spacing)
    if xe.size < 3:
        raise ValueError("too few envelope points to fit a power law")
    lx, ly = np.log(xe), np.log(ve)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (intercept + slope * lx)
    return FitResult(
        "power(env)",
        {"A": float(np.exp(intercept)), "p": float(-slope)},
        _rms(resid),
        xe.size,
    )


def fit_exponential_envelope(lengths, values, min_spacing: float = 0.0) -> FitResult:
    """Fit ``A L^-1/2 exp(-L/xi)`` to the envelope (linear in L and log L)."""
    xe, ve = envelope(lengths, values, min_spacing)
    if xe.size < 3:
        raise ValueError("too few envelope points to fit an exponential")
    y = np.log(ve) + 0.5 * np.log(xe)
    slope, intercept = np.polyfit(xe, y, 1)
    resid = y - (intercept + slope * xe)
    return FitResult(
        "exp(env)",
        {"A": float(np.exp(intercept)),
         "xi_M": float(-1.0 / slope) if slope < 0 else np.inf},
        _rms(resid),
        xe.size,
    )


def fit_crossover(lengths, values, xi_fixed: float, min_spacing: float = 0.0
                  ) -> FitResult:
    """Fit ``A L^-p exp(-L / xi_fixed)`` to the envelope with ``xi`` held fixed.

    This is the estimator that answers the actual question.  Both candidate
    laws share the algebraic prefactor -- a gapped two-dimensional pair decays
    as ``L^-1/2 exp(-L/xi_M)``, and a nodal one as ``L^-p`` with the exponential
    absent -- so fitting a free length and a free exponent at once lets them
    trade against each other over any finite range.  Fixing ``xi`` at the
    independently known ``v_F(theta)/Delta(theta)`` removes that freedom: what
    is left for ``p`` to absorb is precisely the part of the decay that the
    directional gap does not explain.

    On a fully gapped background the fit should return ``p ~ 1/2``, the
    two-dimensional geometric factor.  Anything systematically different, in a
    direction where the gap has a node, is the nodal channel.

    The fit is linear in ``(log A, p)``, so it has no starting-point problem.
    """
    xe, ve = envelope(lengths, values, min_spacing)
    if xe.size < 3:
        raise ValueError("too few envelope points to fit the crossover form")
    # log v + L/xi  =  log A - p log L
    y = np.log(ve) + xe / xi_fixed
    slope, intercept = np.polyfit(np.log(xe), y, 1)
    resid = y - (intercept + slope * np.log(xe))
    return FitResult(
        "crossover",
        {"A": float(np.exp(intercept)), "p": float(-slope),
         "xi_fixed": float(xi_fixed)},
        _rms(resid),
        xe.size,
    )


def discriminate(lengths, values, min_spacing: float = 0.0) -> dict:
    """Decide exponential vs power law on the envelope, and say how safely.

    Over a narrow range in ``L`` the two laws are nearly degenerate -- an
    exponential over a factor of three in ``L`` is fitted almost as well by some
    power law -- so the verdict is reported together with the decade span and
    the range ratio that produced it.  A verdict from less than about a factor
    of four in ``L`` should not be trusted, and is flagged as such.
    """
    xe, ve = envelope(lengths, values, min_spacing)
    fe = fit_exponential_envelope(lengths, values, min_spacing)
    fp = fit_power_envelope(lengths, values, min_spacing)
    span = float(np.log10(ve.max() / ve.min())) if ve.min() > 0 else np.inf
    ratio = float(xe.max() / xe.min())
    return {
        "exponential": fe,
        "power": fp,
        "verdict": "exponential" if fe.rms_log < fp.rms_log else "power law",
        "rms_ratio": float(fp.rms_log / fe.rms_log) if fe.rms_log > 0 else np.inf,
        "decades": span,
        "L_ratio": ratio,
        "decisive": ratio >= 4.0 and abs(fe.rms_log - fp.rms_log)
        > 0.25 * max(fe.rms_log, fp.rms_log),
    }


def window_stability(
    lengths, values, min_spacing: float = 0.0, n_windows: int = 6,
    min_points: int = 5
) -> tuple[float, float, list[tuple[float, float, float]]]:
    """Refit the envelope exponent over sliding windows in L.

    Returns ``(p_mean, p_spread, per_window)`` where ``per_window`` lists
    ``(L_min, L_max, p)``.  The spread is the half-range of the fitted
    exponents; it is the number quoted as the uncertainty on ``p``, because a
    power law measured over a finite range is limited by drift with the fitting
    window rather than by noise on the individual points.
    """
    xe, ve = envelope(lengths, values, min_spacing)
    if xe.size < min_points + 1:
        raise ValueError("not enough envelope points for a stability scan")
    out: list[tuple[float, float, float]] = []
    total = xe.size
    span = max(min_points, total - n_windows + 1)
    for start in range(0, total - span + 1):
        sl = slice(start, start + span)
        lx, ly = np.log(xe[sl]), np.log(ve[sl])
        slope, _ = np.polyfit(lx, ly, 1)
        out.append((float(xe[sl][0]), float(xe[sl][-1]), float(-slope)))
    ps = np.array([o[2] for o in out])
    return float(np.mean(ps)), float(0.5 * (ps.max() - ps.min())), out
