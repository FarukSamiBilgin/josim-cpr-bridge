# `nodal_majorana` — vortex–vortex Majorana hybridization on a nodal superconductor

A self-contained Bogoliubov–de Gennes calculation of how two vortex-bound
Majorana modes split as a function of their separation `L`, on a background
whose superconducting gap has **nodes**.

> **Scope note.** This module is independent of the JoSIM CPR bridge that the
> rest of this repository implements. It shares the repository, not the code:
> nothing here imports `josim_bridge`, and nothing there imports this.

---

## The question

In a fully gapped topological superconductor the splitting between two
vortex-bound Majorana modes is textbook:

```
|δE(L)| = A · L^(-1/2) · exp(-L / ξ_M) · |cos(k_F L + φ)|
```

Exponential decay, an oscillation at the Fermi wavelength, and the `L^(-1/2)`
of two-dimensional propagation. On a **nodal** background the mediating
quasiparticles are gapless along the nodal directions, so the exponential has
nothing to decay against, and the natural guess is that it is replaced by a
power law `|δE| ~ L^(-p)`.

The exponent `p` is the number this module set out to measure.

## What it found instead

**The power law is not an observable of a two-level system**, and the
calculation says so with numbers rather than by failing to converge.

Two energy scales control the problem, and on a nodal background *both* track
the gap at the node, `Δ_n`, rather than the maximum gap:

| scale | meaning | size |
|:--|:--|:--|
| `ξ_M(θ) = v_F(θ) / Δ(θ)` | hybridization length along the vortex axis | diverges as `Δ_n → 0` |
| `E₁` | Caroli–de Gennes–Matricon core-level minigap | `≈ 0.24 Δ_n` (measured) |

A pair of Majorana levels *exists as a pair* only while `|δE| < E₁`; above that
the two states are core-ladder states, not a Majorana doublet. Since `|δE|`
itself is of order `Δ_n` at short separations, that condition bites exactly in
the regime `L ≪ ξ_M` where the power law would be visible. Sending `Δ_n → 0`
opens the power-law window and closes the minigap at the same rate, so the two
never comfortably overlap.

The endpoint of that argument is directly visible: **on the bare nodal
background a vortex binds no zero mode at all.** Its lowest core-weighted level
does not converge as the box grows, and its core weight stays around `0.13–0.21`
— it is a resonance inside the nodal continuum, not a bound state. The
"splitting" one would read off from it is a property of the box.

So the deliverable is the law that *does* govern the accessible regime.

## The result: the anisotropic hybridization law

Regulating the nodes with a `d + is` component of amplitude `d_is` (which opens
a gap `≈ 0.69 d_is` at the nodes, leaves the antinodal gap essentially
untouched, and makes the Chern number odd — so a genuine bound Majorana exists)
lets `δE(L)` be measured cleanly. Across four regulator strengths and three
directions of the vortex–vortex axis, the measured decay length obeys

```
ξ_M(θ) = v_F(θ) / Δ(θ)
```

with `v_F(θ)` and `Δ(θ)` read off the **clean band structure** along the same
ray in momentum space — values the fit is never shown. See
`results/law.json` and the summary table printed by `run_exponent.py`.

The device-relevant consequence: the vortex separation needed to suppress
hybridization is set by the gap **in the direction of the vortex axis**, not by
the maximum gap. On a nodal background the required separation therefore
diverges as the nodal direction is approached, and the anisotropy ratio between
the worst and best axes is `Δ_max / Δ_n`.

---

## Model

A single-band square-lattice metal with Rashba spin–orbit coupling and an
out-of-plane Zeeman field, proximitized by a spin-singlet pair potential:

```
h(k)  = ξ(k) σ₀ + α (sin k_y σ_x − sin k_x σ_y) + V_z σ_z
ξ(k)  = 2t (2 − cos k_x − cos k_y) − μ
H_BdG = [[ h(k), Δ(k) iσ_y ], [ (Δ(k) iσ_y)†, −h(−k)* ]]
```

The pairing form factor is a free choice:

| `form` | `Δ(k)` | character |
|:--|:--|:--|
| `s` | `Δ₀` | fully gapped, Chern `−1` — the benchmark |
| `ext-s` | `Δ₀ [d_s + (cos k_x + cos k_y)/2]` | tunable from gapped to nodal |
| `d` | `Δ₀ (cos k_x − cos k_y)/2` | nodes on the zone diagonals |

plus an optional on-site `i·d_is` component — the **node regulator** described
above.

Vortices multiply every pair-potential matrix element, at its bond midpoint, by

```
Δ → Δ · Π_v tanh(|r − R_v| / ξ₀) · Π_v (r − R_v)/|r − R_v|
```

Writing the phase as a product of unit complex numbers keeps it single-valued —
no branch cut is ever crossed — so open boundary conditions are consistent for
any number of vortices. Only the pair potential carries the vortex; the vector
potential is dropped (the standard extreme type-II / London-limit treatment,
valid when the magnetic length far exceeds both the coherence length and the
separations scanned).

## Method, and the three things that make it trustworthy

**1 · The real-space assembly is pinned to the Bloch Hamiltonian.**
`tests/test_model.py` builds the Hamiltonian in real space with periodic
boundaries, Fourier transforms it back, and demands agreement with an
independently written 4×4 Bloch matrix at every allowed momentum, for every
form factor and for the regulator. Agreement is at `4 × 10⁻¹⁵`. Every sign,
factor of two and `1/(2i)` in the correspondence is fixed by that one test.

**2 · The vortex states are identified by core weight, then by energy.**
On a finite open lattice the states nearest zero are generally *not* the
vortex-derived ones — a chiral topological superconductor carries a gapless
Majorana edge mode, and a nodal one carries a gapless bulk continuum. Reading
off the two smallest eigenvalues measures the box. So states are filtered by
their weight within `3ξ₀` of a core (a fixed radius, not one that grows with
`L`, or delocalized states would creep past the threshold at large separations)
and the Majorana pair is then the core-localized pair *closest to zero* — the
CdGM ladder carries the larger core weight, so selecting by weight alone picks
the wrong states.

**3 · Every point is converged in box size before it is used.**
`converged_splitting` grows the clearance between the vortices and the boundary
until `δE` stops moving, and returns nothing when it never does. Points that
fail are reported and dropped, not fitted.

Fits are done on the **envelope** (the peaks, where the oscillation factor is
one) as well as on the full oscillating series, and the two must agree. The
envelope uses neighbour-based peak-picking, not a sliding window: because the
signal decays, a window as wide as one oscillation period rejects every peak
but the first.

## Validation

`run_benchmarks.py` — checks in limits where the answer is already known:

- The gapped `s`-wave background has Chern number `−1`, so a single vortex binds
  one Majorana; under open boundaries it pairs with the sample edge, and that
  splitting falls exponentially with the box (`1.4 × 10⁻²` → `3.6 × 10⁻⁵` over
  boxes 24→64), staying far below the CdGM minigap.
- Two vortices in that background reproduce the textbook exponential law, and
  the fitted `ξ_M` and `k_F` match `v_F/Δ_gap` and the Fermi surface **to within
  10%** — neither reference value is ever shown to the fit.
- The envelope prefers the exponential over a power law, and the run reports
  whether the `L` range was wide enough for that preference to mean anything.
  It is: the scan spans about five decades of splitting.

`tests/test_fits.py` — the fitting layer against synthetic series with known
answers: it recovers `ξ_M` to 0.06%, `k_F` to 0.02% and `p` to 0.1%, picks the
correct law in both directions, and flags a deliberately narrow scan as
indecisive.

`tests/test_model.py` — Bloch consistency (above), hermiticity, particle–hole
symmetry of the spectrum, vortex winding numbers (`+1` about one core, `+2`
about both), and the collapse of the `d`-wave gap under mesh refinement.

## Reproducing

```bash
pip install numpy scipy

python3 nodal_majorana/tests/test_model.py     # conventions and symmetries
python3 nodal_majorana/tests/test_fits.py      # the fitting layer
python3 nodal_majorana/run_benchmarks.py       # known-limit benchmarks
python3 nodal_majorana/run_exponent.py         # the production run
python3 nodal_majorana/run_exponent.py --quick # ~10x faster, coarser grid
```

The production run writes `nodal_majorana/results/law.json` with every measured
point, every fit and every dropped separation.

## Files

| File | Role |
|:--|:--|
| `model.py` | Parameters, `k`-space and real-space BdG, vortex profiles, Chern number, Fermi surface |
| `solve.py` | Sparse shift–invert eigensolve, core-weight mode identification, box-size convergence |
| `fits.py` | Exponential and power-law fits, envelope extraction, law discrimination, window stability |
| `run_benchmarks.py` | Checks in the gapped limit where the answer is known |
| `run_exponent.py` | The production run: nodal structure, no-bound-state result, the hybridization law |
| `tests/` | Convention, symmetry and fitting-layer tests |

## Honest limits

- **`p` is not measured here.** The reason is given as an inequality between
  `|δE(L)|` and the CdGM minigap, both of which are measured, rather than as a
  fitting failure. Reaching the power-law regime needs a window
  `ξ₀ ≪ L ≪ ξ_M` that is simultaneously inside the minigap; the two conditions
  pull against each other as the nodes are approached.
- **`ξ₀` is a fixed `tanh` profile, not self-consistent.** The pair potential is
  imposed, not solved for. A self-consistent core would change the CdGM ladder
  by an O(1) factor and so shift the numbers, but not the direction of the
  inequality.
- **The vector potential is dropped** (London limit). Superflow around the
  vortices Doppler-shifts the nodal quasiparticles; on a nodal background that
  shift is not parametrically small, so it should be expected to change
  prefactors.
- **One model, one normal state.** The `ξ_M = v_F/Δ` law is checked across four
  regulator strengths and three axis directions within a single band structure.
  The functional form is standard; the point of the check is that the
  *anisotropy* follows the directional gap quantitatively.
