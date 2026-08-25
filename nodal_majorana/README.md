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

## What it found

**1 · On the bare nodal background a vortex binds no zero mode.**
Its lowest core-weighted level does not converge as the lattice grows — it
wanders over `1.3 × 10⁻²` … `9.1 × 10⁻²` across boxes 32→72 — and its core
weight stays at `0.13–0.25`. In the same boxes the gapped `s`-wave Majorana
falls monotonically from `2.9 × 10⁻⁴` to `2.3 × 10⁻⁶` at core weight `0.485`.
The nodal "Majorana" is a resonance inside the gapless continuum, and a
splitting read off from it would be a property of the box.

**2 · With the nodes regulated, the hybridization length follows the gap along
the vortex axis.**

```
ξ_M(θ) = c · v_F(θ) / Δ(θ),        c = 1.11 ± 0.11
```

`v_F(θ)` and `Δ(θ)` are read off the clean band structure along the same ray in
momentum space — values the fit is never shown. Across the four regulator
strengths and both axes that pass the quality cut, the measured `ξ_M` sits at
`1.02–1.23` times the prediction:

| node gap `Δ_n` | `ξ_fit/ξ_ref`, antinodal (0°) | `ξ_fit/ξ_ref`, nodal (45°) |
|--:|--:|--:|
| 0.684 | 1.05 | 1.02 |
| 0.479 | 1.22 | 1.19 |
| 0.342 | 1.06 | 1.23 |
| 0.274 | 1.41 * | 3.93 * |

`*` fails the quality cut — see below.

The anisotropy is the sharper test, because it is a ratio of two measurements
in the same model and the calibration `c` cancels out of it entirely:

| node gap `Δ_n` | measured `ξ_M(45°)/ξ_M(0°)` | predicted from the gaps |
|--:|--:|--:|
| 0.684 | 1.46 | 1.50 |
| 0.479 | 1.87 | 1.90 |
| 0.342 | 2.88 | 2.49 |
| 0.274 | 8.46 * | 3.02 * |

Agreement to 2%, 2% and 16% on the three usable points, over a factor of two in
the anisotropy itself.

The consequence that matters for a device: **the separation needed to suppress
hybridization is set by the gap in the direction of the vortex axis, not by the
maximum gap.** On a nodal background that separation therefore grows without
bound as the axis turns toward a node — which is the practical content of the
"hybridization law" for a nodal superconductor, whatever `p` turns out to be.

**3 · The exponent `p` is not resolved, and the obstruction is measured rather
than guessed.**
Fixing the length at the calibrated `c · v_F/Δ` and fitting the algebraic
remainder `|δE| = A L^(-p) exp(-L/ξ_M)` gives `p = 0.48 ± 0.34` on the antinodal
axis (consistent with the `1/2` of two-dimensional propagation) and
`0.16 ± 0.21` … `0.69 ± 0.33` on the nodal axis. The nodal values trend low but
the bands overlap: the ±0.11 uncertainty in `c` alone propagates to ±0.2–0.5 in
`p`, which is larger than the effect being looked for.

Pushing to a weaker node gap does not help, and the reason is specific. The pair
is a resolvable two-level system only while `|δE|` stays inside the
Caroli–de Gennes–Matricon core-level minigap, and that minigap tracks the *node*
gap (`E₁ ≈ 0.24 Δ_n`, measured). At the weakest regulator, 17 of 42 separations
fail the convergence and resolvability filters — **and every one of them lies at
`L ≤ 15`, exactly the sub-`ξ_M` window where the power law would live.** What
survives spans `0.70` decades over a factor `8.2` in `L`, well below the
threshold this package's own discrimination test requires. So the power-law
regime is squeezed from both sides as `Δ_n → 0`: the window widens, and the
minigap that makes it observable closes at the same rate.

**4 · A methodological warning worth keeping.**
Measured over `L ∈ [4, 20]`, the nodal scan at `Δ_n = 0.342` gave
`ξ_fit/ξ_ref = 3.75` and looked convincingly non-exponential. Measured over
`L ∈ [3, 32]` — the same model, the same axis — it gives `1.23`. The first
window simply never reached the exponential tail. Over a narrow range in `L` an
exponential and a power law are close to degenerate, which is why every fit here
is reported with the decade span and the `L` ratio that produced it, and why the
benchmark checks that its own verdict is decisive before believing it.

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
python3 nodal_majorana/run_exponent.py         # the production run   (~25 min)
python3 nodal_majorana/run_exponent.py --deep  # weaker regulators, wider L
python3 nodal_majorana/run_exponent.py --quick # ~10x faster, coarser grid

python3 nodal_majorana/analyse_law.py                        # analyse law.json
python3 nodal_majorana/analyse_law.py nodal_majorana/results/law_deep.json
```

The production runs write `results/law.json` and `results/law_deep.json` with
every measured point, every fit and every dropped separation; the analysis
writes `*_analysis.json` beside them. The numbers quoted above come from the
`--deep` run, which is the one whose `L` range reaches the exponential tail.

## Files

| File | Role |
|:--|:--|
| `model.py` | Parameters, `k`-space and real-space BdG, vortex profiles, Chern number, Fermi surface |
| `solve.py` | Sparse shift–invert eigensolve, core-weight mode identification, box-size convergence |
| `fits.py` | Exponential and power-law fits, envelope extraction, law discrimination, window stability |
| `run_benchmarks.py` | Checks in the gapped limit where the answer is known |
| `run_exponent.py` | The production run: nodal structure, no-bound-state result, `δE(L)` scans (`--deep` for weaker regulators and a wider `L` range) |
| `analyse_law.py` | Post-processing: per-scan decay law, the calibration of `c`, the algebraic exponent, and the quality cut |
| `results/` | Every measured point, fit and dropped separation, as JSON |
| `tests/` | Convention, symmetry and fitting-layer tests |

## Honest limits

- **`p` is not measured here.** What is measured is why: the algebraic exponent
  comes out `0.16–0.69` on the nodal axis against `0.48 ± 0.34` on the antinodal
  one, bands that overlap because the ±0.11 uncertainty in the calibration alone
  costs ±0.2–0.5 in `p`; and pushing to a weaker node gap loses the sub-`ξ_M`
  separations to the resolvability condition faster than it widens the window.
  A sharper answer needs a model where the core-level minigap is a larger
  fraction of the node gap, or lattices several times larger than the ones used
  here — not more fitting.
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
