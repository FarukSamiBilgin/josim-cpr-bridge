# Theory — from a current–phase relation to a qubit

This note derives, end to end, how the bridge turns a junction's current–phase
relation (CPR) into a qubit spectrum, and why a *skewed* CPR makes the qubit
frequency and anharmonicity **tunable**. Every numerical claim here is checked by
[`validation/test_physics.py`](../validation/test_physics.py).

---

## 1. The CPR sets the potential

A Josephson element stores energy equal to the integral of its supercurrent over
phase:

$$
U(\varphi) \;=\; \frac{\hbar}{2e}\int_0^{\varphi} I(\varphi')\, d\varphi'.
$$

So the CPR `I(φ)` *is* the junction — pick the CPR and the potential follows. The
textbook tunnel junction `I(\varphi)=I_c\sin\varphi` integrates to the familiar
cosine well:

$$
U(\varphi) = -E_J\cos\varphi, \qquad E_J = \frac{\hbar I_c}{2e}.
$$

Every circuit‑quantization tool that "assumes a cosine" is really assuming this one
CPR. The contribution of this project is to let `I(φ)` be **anything** and keep the
bookkeeping consistent with JoSIM.

---

## 2. Two CPR parametrizations that cover the interesting cases

### Harmonic (general)

Any `2π`‑periodic, odd CPR is a sine series — exactly JoSIM's `cpr={…}` vector:

$$
I(\varphi) = I_c\sum_{n\ge 1} c_n \sin(n\varphi)
\quad\Longrightarrow\quad
U(\varphi) = -E_J\sum_{n\ge 1}\frac{c_n}{n}\cos(n\varphi).
$$

A forward skew is just a few extra harmonics with the signs that steepen the rise
and push the peak past `π/2`.

### Transparency (short ballistic weak link)

For a short channel of transparency `τ ∈ (0,1]`, the Andreev bound states give the
Beenakker / Furusaki–Tsukada CPR:

$$
I(\varphi) = I_c\,\frac{\sin\varphi}{\sqrt{1-\tau\sin^2(\varphi/2)}}.
$$

This is the canonical graphene / Dirac / point‑contact form. As `τ → 0` it reduces
to `sin φ`; as `τ → 1` it sharpens toward a sawtooth. Its potential is the Andreev
ground‑state energy,

$$
U(\varphi) \;\propto\; -\sqrt{1-\tau\sin^2(\varphi/2)},
$$

the tilted well drawn in the banner and pipeline figures.

> **Useful fact (verified on JoSIM v2.7).** JoSIM's temperature‑dependent Haberkorn
> branch — switched on by the `D` parameter — evaluates exactly this transparency
> CPR, and at `T → 0` reduces to the Beenakker form with `τ = D`. So the canonical
> graphene CPR is already representable in stock JoSIM.

---

## 3. The qubit Hamiltonian

Add charging energy and quantize the phase. In the charge basis,

$$
H = 4E_C\,\hat n^2 + U(\hat\varphi),
\qquad
E_C = \frac{e^2}{2C},
$$

with `[\hat\varphi,\hat n] = i`. The bridge builds `H` on a charge grid and
diagonalizes it; the transition frequencies are `ω_{ij} = E_j - E_i`. An
**independent** phase‑basis finite‑difference solver is used in the test suite as a
cross‑check — the two agree to `< 10^{-3}` MHz.

In the transmon regime (`E_J/E_C ≫ 1`) the cosine results are the familiar

$$
\omega_{01}\approx\sqrt{8E_JE_C}-E_C,
\qquad
\alpha \equiv \omega_{12}-\omega_{01}\approx -E_C .
$$

Both shift once the CPR — hence the well — is skewed.

---

## 4. Where the tunability comes from: the well's Taylor coefficients

Expand the potential about its minimum `φ_min`:

$$
U(\varphi) \approx U_0 + E_J\Big[c_2(\varphi-\varphi_{\min})^2 + c_3(\varphi-\varphi_{\min})^3 + c_4(\varphi-\varphi_{\min})^4 + \dots\Big].
$$

`taylor_coeffs(...)` returns exactly `{c2, c3, c4, phi_min}`. Their meaning:

| Coefficient | Sets | Physical knob |
|:--|:--|:--|
| `c₂` | curvature → `ω₀₁` | qubit frequency |
| `c₃` | cubic → `g₃` | **three‑wave mixing** |
| `c₄` | quartic → `K` | **Kerr / anharmonicity** |

For the cosine well, `U = -E_J\cos\varphi` expands to `c_2 = 1/2`, `c_4 = -1/24`,
so the **Kerr‑per‑unit‑frequency ratio is fixed**:

$$
\left.\frac{c_4}{c_2}\right|_{\cos} = \frac{-1/24}{1/2} = -\frac{1}{12}.
$$

This is why a conventional transmon's anharmonicity tracks `E_C` and nothing else.

### The result

For the transparency CPR, the same expansion (carried out symbolically and verified
numerically to `5.7\times10^{-6}`) gives a ratio that **moves with transparency**:

$$
\boxed{\;\frac{c_4}{c_2} = \frac{\tau}{16} - \frac{1}{12}\;}
$$

| `τ` | `c₄/c₂` | meaning |
|:--:|:--:|:--|
| `0` (cosine) | `-0.0833` | the locked transmon value |
| `0.5` | `-0.0521` | |
| `0.9` | `-0.0271` | ~3× smaller \|α\| at fixed `E_C` |

Concretely, at `E_C = 0.25`, `E_J = 12.5` GHz the bridge returns
`α = -287.3 MHz` for the cosine and `α = -96.0 MHz` for `τ = 0.9` — the same `E_C`,
a different qubit. **Anharmonicity has become a design parameter.**

A finite `c₄` change also shifts the optimal point for higher‑order cancellations;
at special bias the effective Kerr can be tuned through zero (a *Kerr‑free* point),
which is desirable for parametric and three‑wave devices.

---

## 5. Three‑wave mixing needs broken symmetry

`g₃ ∝ c₃`, and `c₃ = 0` whenever the well is symmetric about its minimum — which is
the case for the bare CPRs above (`phi_min = 0`). The test suite confirms `c₃ ≈ 0`
to `~10^{-13}`. To turn it on you break the symmetry:

- a **flux bias** that moves `φ_min` off zero, or
- an **asymmetric loop** (SNAIL‑type) whose CPR is not odd about its minimum.

The bridge supports a phase offset for exactly this exploration. Practically:
combine a skewed `c₄` (transparency) with a bias‑induced `c₃` to target a chosen
`(g₃, K)` pair.

---

## 6. A coherence proxy: T₁ and the matrix element

With a standard dielectric‑loss channel (loss tangent `tan δ`), the relaxation rate
follows Fermi's golden rule,

$$
\frac{1}{T_1}\;\propto\; |\langle 0|\hat n|1\rangle|^2\,\omega_{01}\,S(\omega_{01}),
$$

so the charge matrix element `|⟨0|n̂|1⟩|²` is the lever the CPR pulls. The bridge
returns, at `E_C=0.25`, `E_J=12.5` GHz, `tan δ = 10^{-6}`:

| Junction | `T₁` |
|:--|:--:|
| cosine | `67.2 µs` |
| `τ = 0.9` | `78.8 µs` |

`T₁` *improves* modestly with transparency because the matrix element drops. The
**absolute** value depends on the assumed noise spectrum; the **ratio** to the
cosine and the matrix element itself are the robust, model‑independent outputs.

---

## 7. Mapping to real graphene / TBG junctions

The transparency `τ` and critical current `I_c` are not free — they are set by the
material and the gate:

- **`τ(V_g)`** — near the Dirac point the channel is poorly transmitting (`τ` small,
  nearly sinusoidal CPR); away from it the ballistic channels open (`τ → 1`, strong
  skew). The bridge's `graphene_cpr.tau_of_vg` is a smooth phenomenological stand‑in
  for this.
- **`I_c(V_g) ∝ √(V_g)`** — the critical current grows with carrier density; the
  reference `ic_of_vg` uses this `√` scaling, reproduced in the gate table of the
  getting‑started guide.
- **Twisted bilayer graphene** adds flat‑band physics, making both `τ` and `I_c`
  exceptionally gate‑sensitive — i.e. a qubit whose `ω₀₁` and `α` are tunable *in
  situ*, which is the practical motivation for the whole pipeline.

The roadmap item is to validate `τ(V_g)` against a digitized, measured ballistic CPR
and reproduce the published skewness‑vs‑gate trend.

---

## References

1. W. Haberkorn, H. Knauer, J. Richter, *A theoretical study of the current–phase
   relation in Josephson contacts*, **Phys. Status Solidi A 47**, K161 (1978).
2. C. W. J. Beenakker, *Universal limit of critical‑current fluctuations in
   mesoscopic Josephson junctions*, **Phys. Rev. Lett. 67**, 3836 (1991).
3. G. Nanda *et al.*, *Current–phase relation of ballistic graphene Josephson
   junctions*, **Nano Lett. 17**, 3396 (2017).
4. M. T. English *et al.*, *Observation of nonsinusoidal current–phase relation in
   graphene Josephson junctions*, **Phys. Rev. B 94**, 115435 (2016).
5. J. Koch *et al.*, *Charge‑insensitive qubit design derived from the Cooper‑pair
   box* (the transmon), **Phys. Rev. A 76**, 042319 (2007).

---

<div align="center">
<sub><a href="GETTING_STARTED.md">← Getting started</a> &nbsp;·&nbsp; <a href="ARCHITECTURE.md">Architecture →</a></sub>
</div>
