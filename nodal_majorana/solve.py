"""Extraction of the vortex-pair splitting delta-E(L) from the BdG spectrum.

The difficulty this module exists to solve
-----------------------------------------
On a finite open lattice the states nearest zero energy are *not* generally the
vortex-derived ones.  A chiral topological superconductor carries a gapless
Majorana edge mode, whose finite-size ladder has spacing ~ 2 pi v_edge / P; a
nodal superconductor carries a gapless bulk continuum whose finite-size spacing
is ~ v_Delta / L_sys.  Either can sit closer to zero than the pair splitting we
want.  Simply reading off the two smallest eigenvalues therefore measures the
box, not the physics.

The fix used here is to identify the vortex-derived states by their weight on
the vortex cores.  Edge and continuum states have core weight smaller by orders
of magnitude, so the selection is unambiguous whenever the vortices are not
themselves touching the boundary -- and `SplittingResult` reports the separation
between accepted and rejected weights so that this can be checked rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .model import Lattice, Model, VortexConfig, build_hamiltonian, two_vortices


@dataclass
class SplittingResult:
    """Outcome of one delta-E(L) measurement, with its own diagnostics."""

    sep: float
    angle_deg: float
    lx: int
    ly: int
    delta_e: float          # E_+ - E_-  for the selected vortex pair
    energies: np.ndarray    # all computed eigenvalues, sorted
    core_weights: np.ndarray  # matching core weights
    selected: tuple[int, int]
    w_selected: float       # smallest core weight among the selected pair
    minigap: float          # |E| of the next core-localized level (CdGM)
    e_next: float           # |E| of the lowest state not selected (any character)

    @property
    def clean(self) -> bool:
        """Whether the selected pair really is the in-gap Majorana pair.

        Two conditions: the states must be genuinely core-localized, and they
        must sit well inside the Caroli-de Gennes-Matricon minigap.  Neither
        condition speaks to finite-size contamination from the boundary -- that
        is what :func:`converged_splitting` is for.
        """
        return self.w_selected > 0.30 and abs(self.delta_e) < 0.5 * self.minigap

    @property
    def edge_distance(self) -> float:
        """Distance from either vortex core to the nearest sample edge."""
        ang = np.deg2rad(self.angle_deg)
        dx, dy = 0.5 * self.sep * abs(np.cos(ang)), 0.5 * self.sep * abs(np.sin(ang))
        return min((self.lx - 1) / 2.0 - dx, (self.ly - 1) / 2.0 - dy)


def core_weights(
    vecs: np.ndarray, lat: Lattice, centers, radius: float
) -> np.ndarray:
    """Fraction of each eigenvector's norm within ``radius`` of any vortex core."""
    mask = np.zeros(lat.n, dtype=bool)
    for cx, cy in centers:
        mask |= np.hypot(lat.xs - cx, lat.ys - cy) <= radius
    site_of_row = np.repeat(np.arange(lat.n), 4)
    dens = np.abs(vecs) ** 2                       # (4N, nev)
    tot = dens.sum(axis=0)
    inside = dens[mask[site_of_row], :].sum(axis=0)
    return inside / np.where(tot > 0, tot, 1.0)


def lowest_states(h: sp.spmatrix, nev: int = 24, sigma: float = 0.0):
    """Eigenpairs nearest ``sigma`` by sparse shift-invert.

    A small real shift is applied if the factorization hits the exact zero of an
    (accidentally) singular matrix.
    """
    for shift in (sigma, sigma + 1e-9, sigma + 1e-7, sigma + 1e-5):
        try:
            vals, vecs = spla.eigsh(h, k=nev, sigma=shift, which="LM")
            order = np.argsort(vals)
            return vals[order], vecs[:, order]
        except (RuntimeError, spla.ArpackNoConvergence, ValueError):
            continue
    raise RuntimeError("shift-invert eigensolver failed to converge")


def _candidates(vals: np.ndarray, w: np.ndarray, w_min: float) -> np.ndarray:
    """Indices of core-localized states, ordered by |E|.

    Two families sit on the cores: the Majorana pair at +/- delta_e/2 and the
    Caroli-de Gennes-Matricon ladder at the minigap and above.  The CdGM states
    typically carry the *larger* core weight, so selecting by weight alone picks
    the wrong states; the Majorana pair is the core-localized pair closest to
    zero.  Hence: threshold on weight, then order by |E|.
    """
    cand = np.nonzero(w > w_min)[0]
    if cand.size < 2:
        cand = np.argsort(w)[-2:]
    return cand[np.argsort(np.abs(vals[cand]))]


def _select_pair(vals: np.ndarray, w: np.ndarray, w_min: float) -> tuple[int, int]:
    """Pick the vortex-derived +/- pair: core-localized, then closest to zero."""
    order = _candidates(vals, w, w_min)
    # particle-hole symmetry demands one state either side of zero
    pos = [i for i in order if vals[i] >= 0]
    neg = [i for i in order if vals[i] < 0]
    if pos and neg:
        return int(neg[0]), int(pos[0])
    return int(order[0]), int(order[1])


def measure_splitting(
    model: Model,
    lx: int,
    ly: int,
    sep: float,
    angle_deg: float,
    nev: int = 24,
    core_radius: float | None = None,
    mu_map: np.ndarray | None = None,
    w_min: float = 0.15,
) -> SplittingResult:
    """Build the two-vortex Hamiltonian and extract ``delta_e = E_+ - E_-``."""
    cfg = two_vortices(lx, ly, sep, angle_deg)
    h = build_hamiltonian(model, lx, ly, cfg, mu_map=mu_map)
    lat = Lattice(lx, ly)
    vals, vecs = lowest_states(h, nev=nev)
    # A fixed radius, not one that grows with the separation: a core disk that
    # scales with L would let delocalized states accumulate enough weight to
    # pass the selection threshold at large separations.
    rad = core_radius if core_radius is not None else 3.0 * model.xi0
    w = core_weights(vecs, lat, cfg.centers, rad)

    i_neg, i_pos = _select_pair(vals, w, w_min)
    sel = {i_neg, i_pos}
    rest = [i for i in range(len(vals)) if i not in sel]
    core_rest = [i for i in rest if w[i] > w_min]
    e_next = float(min(abs(vals[i]) for i in rest)) if rest else np.inf
    # If no other core-localized level was computed, the next one lies beyond the
    # window that was diagonalized; the edge of that window is a lower bound on
    # the minigap, which is all the `clean` test needs.
    minigap = (
        float(min(abs(vals[i]) for i in core_rest))
        if core_rest
        else float(np.max(np.abs(vals)))
    )

    return SplittingResult(
        sep=sep,
        angle_deg=angle_deg,
        lx=lx,
        ly=ly,
        delta_e=float(vals[i_pos] - vals[i_neg]),
        energies=vals,
        core_weights=w,
        selected=(i_neg, i_pos),
        w_selected=float(min(w[i_neg], w[i_pos])),
        minigap=minigap,
        e_next=e_next,
    )


def converged_splitting(
    model: Model,
    sep: float,
    angle_deg: float,
    pads: tuple[int, ...] = (16, 24, 32, 40),
    rtol: float = 0.08,
    nev: int = 24,
    w_min: float = 0.15,
    verbose: bool = False,
) -> tuple[SplittingResult | None, list[SplittingResult]]:
    """Grow the box until ``delta_e`` stops moving, and return the first
    size-independent value.

    The lattice is sized as ``sep + 2 * pad`` along each axis, so ``pad`` is the
    clearance between a vortex core and the boundary.  Because the vortex
    Majoranas also hybridize with the sample edge, every splitting measured in a
    finite box has a floor of order ``exp(-pad / xi_M)``; a point is only
    trustworthy once two successive pads agree.  Returns ``(result, history)``
    with ``result = None`` when no pad in the list converged, which is itself
    the honest answer for that separation.
    """
    history: list[SplittingResult] = []
    prev: SplittingResult | None = None
    for pad in pads:
        side = int(np.ceil(sep)) + 2 * pad
        side += side % 2  # keep the parity fixed so the geometry is comparable
        r = measure_splitting(
            model, side, side, sep, angle_deg, nev=nev, w_min=w_min
        )
        history.append(r)
        if verbose:
            print(
                f"      pad={pad:3d} box={side:3d} dE={r.delta_e:+.6e} "
                f"w={r.w_selected:.3f} clean={r.clean}"
            )
        if prev is not None and r.clean and prev.clean:
            scale = max(abs(r.delta_e), abs(prev.delta_e))
            if scale > 0 and abs(abs(r.delta_e) - abs(prev.delta_e)) / scale < rtol:
                return r, history
        prev = r
    return None, history


def single_vortex_energy(
    model: Model, lx: int, ly: int, nev: int = 24, w_min: float = 0.15
) -> tuple[float, float, float]:
    """Lowest core-localized |E| of a single centred vortex.

    Under open boundaries a lone vortex carries one Majorana and the sample edge
    carries its partner, so this energy is their hybridization and must fall
    exponentially as the lattice grows -- the cheapest check that the zero mode
    exists at all.  Returns ``(|E|, core weight, CdGM minigap)``, the minigap
    being the next core-localized level up, which sets the ceiling on splittings
    this method can resolve.
    """
    cfg = VortexConfig(centers=(((lx - 1) / 2.0, (ly - 1) / 2.0),))
    h = build_hamiltonian(model, lx, ly, cfg)
    lat = Lattice(lx, ly)
    vals, vecs = lowest_states(h, nev=nev)
    w = core_weights(vecs, lat, cfg.centers, 3.0 * model.xi0)
    order = _candidates(vals, w, w_min)
    i = int(order[0])
    minigap = float(abs(vals[order[2]])) if order.size > 2 else float("nan")
    return float(abs(vals[i])), float(w[i]), minigap


def radial_profile(
    model: Model,
    lx: int,
    ly: int,
    angle_deg: float,
    nev: int = 16,
    nbins: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Angle-resolved decay |psi(r)|^2 of a single vortex's lowest core state.

    Samples sites within +/- 12 degrees of the given ray from the core and bins
    them in radius.  Comparing the 0-degree and 45-degree rays of a d-wave
    background shows the anisotropy of the Majorana tail directly, independently
    of the two-vortex fit.
    """
    cx, cy = (lx - 1) / 2.0, (ly - 1) / 2.0
    cfg = VortexConfig(centers=((cx, cy),))
    h = build_hamiltonian(model, lx, ly, cfg)
    lat = Lattice(lx, ly)
    vals, vecs = lowest_states(h, nev=nev)
    w = core_weights(vecs, lat, cfg.centers, 3.0 * model.xi0)
    order = _candidates(vals, w, 0.15)
    # sum the +/-E partners: their combination is the Majorana density
    pair = [int(order[0])]
    partner = int(np.argmin(np.abs(vals + vals[order[0]])))
    if partner != pair[0]:
        pair.append(partner)
    dens = sum(
        (np.abs(vecs[:, i]) ** 2).reshape(lat.n, 4).sum(axis=1) for i in pair
    ) / len(pair)

    dx, dy = lat.xs - cx, lat.ys - cy
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    target = angle_deg % 180.0
    dtheta = np.minimum(np.abs(ang - target), 180.0 - np.abs(ang - target))
    sel = (dtheta <= 12.0) & (r > 1.0)

    rmax = 0.5 * min(lx, ly) - 2.0
    edges = np.linspace(1.0, rmax, nbins + 1)
    idx = np.digitize(r[sel], edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    rc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(nbins, np.nan)
    for b in range(nbins):
        m = ok & (idx == b)
        if np.any(m):
            prof[b] = float(np.mean(dens[sel][m]))
    good = np.isfinite(prof) & (prof > 0)
    return rc[good], prof[good]
