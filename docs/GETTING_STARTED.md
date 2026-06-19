# Getting started — from zero to a tunable qubit spectrum

This guide takes you from a fresh clone to running every tool in the repository,
with copy‑paste commands and the **actual output** each one produces. No prior
JoSIM or circuit‑quantization experience is assumed.

> **The one‑sentence model.** A junction's current–phase relation (CPR) defines a
> potential well; the well defines a qubit. This package carries one CPR
> consistently from the *classical* simulator (JoSIM) to the *quantum* spectrum.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Install](#2-install)
3. [Hello, bridge](#3-hello-bridge)
4. [Reading the output](#4-reading-the-output)
5. [Sweeping transparency and gate voltage](#5-sweeping-transparency-and-gate-voltage)
6. [The JoSIM‑side tools](#6-the-josim-side-tools)
7. [Getting a JoSIM binary](#7-getting-a-josim-binary)
8. [Running the validation suites](#8-running-the-validation-suites)
9. [Troubleshooting](#9-troubleshooting)
10. [Where to go next](#10-where-to-go-next)

---

## 1. Prerequisites

| You need | Why | Notes |
|:--|:--|:--|
| **Python ≥ 3.8** | the bridge and tooling are Python | `python --version` |
| **NumPy** | the only third‑party dependency | installed in the next step |
| **A JoSIM binary** *(optional)* | only for the `josim-cli` regression checks and the CPR tracer | see [§7](#7-getting-a-josim-binary) |

The bridge itself (`josim_bridge.py`) and the physics validation are **pure NumPy** —
you can run and verify everything quantum‑side without installing JoSIM at all.

---

## 2. Install

```bash
git clone https://github.com/FarukSamiBilgin/josim-cpr-bridge.git
cd josim-cpr-bridge
pip install -r requirements.txt        # just numpy
```

Smoke‑test the install:

```bash
python validation/test_physics.py
```

Expected — seven checks, all `PASS`:

```text
  [PASS] independent solvers agree (cosine): w01 4.7355/4.7355 GHz, alpha -287.3/-287.3 MHz
  [PASS] independent solvers agree (tau=0.9): w01 4.0419/4.0419 GHz, alpha -96.0/-96.0 MHz
  [PASS] cosine omega01 vs sqrt(8EjEc)-Ec: 4.7355 vs 4.7500 GHz
  [PASS] alpha/EC -> -1 as EJ/EC grows (1/sqrt law)
  [PASS] c4/c2 matches symbolic tau/16-1/12: max err 5.68e-06
  [PASS] pi-shift invariant spectrum: max diff 1.8e-13
  [PASS] D->0 equals cosine: w01 diff 0.000 MHz

ALL PHYSICS CHECKS PASS
```

If you see that, you're ready.

---

## 3. Hello, bridge

Open a Python shell in the repo root and ask the same junction two questions:

```python
import josim_bridge as jb

EC, EJ = 0.25, 12.5          # GHz — a transmon operating point (E_J / E_C = 50)

# 1) the textbook cosine junction
print(jb.qubit_params(EC, EJ, cpr=(1.0,)))

# 2) a high-transparency (τ = 0.9) junction — graphene-like
print(jb.qubit_params(EC, EJ, D=0.9, T=0))
```

Output:

```text
(4.7355, -0.2873)      # cosine:  ω₀₁ = 4.7355 GHz,  α = -287.3 MHz
(4.0419, -0.0960)      # τ = 0.9: ω₀₁ = 4.0419 GHz,  α =  -96.0 MHz
```

Same `E_C`, same `E_J` — **different qubit.** The skew of the CPR dropped the
anharmonicity from −287 MHz to −96 MHz. That is the whole point of the project in
two lines.

---

## 4. Reading the output

`qubit_params` returns `(ω₀₁, α)` in GHz. To see *why* they moved, look at the
Taylor coefficients of the potential well at its minimum:

```python
jb.taylor_coeffs(cpr=(1.0,))     # cosine
jb.taylor_coeffs(D=0.9, T=0)     # τ = 0.9
```

```text
{'c2': 0.50000, 'c3': -1.7e-13, 'c4': -0.04166, 'phi_min': 0.0}   # cosine
{'c2': 0.34098, 'c3':  6.2e-14, 'c4': -0.00923, 'phi_min': 0.0}   # τ = 0.9
```

- `c2` → curvature → sets `ω₀₁`.
- `c4` → quartic term → sets the anharmonicity / **Kerr**.
- `c3` → cubic term → **three‑wave mixing** `g₃`. It is ≈ 0 here because the well is
  symmetric; you unlock it with a flux bias or an asymmetric (SNAIL‑like) loop.

The key ratio is `c4/c2`. For the cosine it is `-0.04166 / 0.5 = -1/12`, **fixed**.
For `τ = 0.9` it is `-0.00923 / 0.341 = τ/16 - 1/12`. That ratio is the design knob.

A coherence estimate from the standard dielectric‑loss channel:

```python
jb.t1_estimate(EC, EJ, cpr=(1.0,), tan_delta=1e-6)   # -> 6.7e-05  (67.2 µs)
jb.t1_estimate(EC, EJ, D=0.9,  T=0, tan_delta=1e-6)   # -> 7.9e-05  (78.8 µs)
```

`T₁` *improves* modestly with transparency, because the charge matrix element
`|⟨0|n|1⟩|²` shrinks. The **absolute** number depends on the noise model; the
**ratio** vs cosine and the matrix element are the robust outputs.

---

## 5. Sweeping transparency and gate voltage

Anharmonicity as a function of transparency:

```python
import numpy as np
for tau in np.linspace(0.0, 0.95, 6):
    w01, a = jb.qubit_params(EC, EJ, D=tau, T=0)
    print(f"τ={tau:0.2f}   ω01={w01:6.3f} GHz   α={a*1000:7.1f} MHz")
```

You will see `α` sweep continuously from ≈ −287 MHz (cosine) toward ≈ −90 MHz as the
junction becomes ballistic — a ~3× tuning range at fixed `E_C`.

To go from "transparency" to "gate voltage", use the graphene reference
(`josim_tools/graphene_cpr.py`), which maps `V_g → (τ, I_c)` — see the next section.

---

## 6. The JoSIM‑side tools

These live in `josim_tools/` and need **no changes to JoSIM** — they produce input
for the *stock* simulator.

### 6.1 Fit any CPR to a JoSIM card — `cpr_to_josim.py`

Turns an arbitrary CPR curve (measurement / DFT / model) into the `cpr={…}`
harmonic vector JoSIM consumes, choosing the harmonic count for a target shape
accuracy.

```bash
cd josim_tools
python cpr_to_josim.py
```

```text
Fitting transparency CPRs to JoSIM cpr={...} (target 1% shape):
  tau=0.6: 3 harmonics, resid=2.53e-03
      .model graphene_tau60 jj(rtype=1 ic=1m cap=50f rn=16 r0=160 cpr={1.0000, -0.1118, 0.0188})
  tau=0.8: 4 harmonics, resid=4.14e-03
      .model graphene_tau80 jj(rtype=1 ic=1m cap=50f rn=16 r0=160 cpr={1.0000, -0.1873, 0.0531, -0.0168})
  tau=0.95: 7 harmonics, resid=6.21e-03
      .model graphene_tau95 jj(rtype=1 ic=1m cap=50f rn=16 r0=160 cpr={1.0000, -0.2989, 0.1377, -0.0714, 0.0391, -0.0221, 0.0127})
```

Use it on your own data:

```python
import numpy as np
from cpr_to_josim import fit_cpr, transparency_cpr, model_line

phi = np.linspace(0, 2*np.pi, 512, endpoint=False)
I   = transparency_cpr(phi, tau=0.85)        # ... or your measured I(φ)
fit = fit_cpr(phi, I, target=1e-2)           # harmonic vector + residual
print(model_line("my_junction", fit))        # ready .model line
```

### 6.2 Gate‑tunable graphene reference — `graphene_cpr.py`

A phenomenological map `τ(V_g)`, `I_c(V_g)` that emits a JoSIM model card per gate
point — the reference implementation for the proposed native C++ path.

```bash
python graphene_cpr.py
```

```text
Gate-tunable graphene CPR reference (static Vg):
    Vg     tau    Ic[mA]  #harm   model card
   0.2   0.176     0.448      2   .model g_vg2  jj(... cpr={1.0000, -0.0241})
   0.5   0.382     0.708      2   .model g_vg5  jj(... cpr={1.0000, -0.0597})
   1.0   0.613     1.000      3   .model g_vg10 jj(... cpr={1.0000, -0.1157, 0.0202})
   2.0   0.839     1.415      4   .model g_vg20 jj(... cpr={1.0000, -0.2083, 0.0659, -0.0233})
   4.0   0.952     2.000      7   .model g_vg40 jj(... cpr={1.0000, -0.3016, 0.1403, -0.0734, ...})
```

Note how both the **skew** (more harmonics) and `I_c` grow away from the Dirac point,
with `I_c ∝ √(V_g)` — the qualitative signature of a ballistic graphene junction.

### 6.3 Ready‑made junction library — `exotic_junctions.lib`

A drop‑in `.include` file with named models for both CPR routes:

```spice
.include josim_tools/exotic_junctions.lib

* harmonic route (works on stock JoSIM today):
*   graphene_tau60 / graphene_tau80 / graphene_tau95
* native transparency route (Haberkorn branch):
*   ballistic_D80 / ballistic_D95
* phase-offset junctions:
*   pi_junction (φ₀ = π)   phi0_junction (anomalous)
```

### 6.4 Trace a CPR with the simulator — `cpr_tracer.cir`

This deck quasi‑statically ramps the junction phase `0 → 2π` and reads the
supercurrent, so the output traces `I(φ)/I_c` for three junctions side by side
(ideal `sin φ`, JoSIM's built‑in `D=0.9` skew, and a 5‑harmonic fit). It needs a
JoSIM binary:

```bash
josim-cli -m -o out.csv josim_decks/cpr_tracer.cir
```

`out.csv` then contains `I(B1)` (sin), `I(B2)` (D=0.9), `I(B3)` (harmonic) versus the
ramped phase — the data behind the validation in [§8](#8-running-the-validation-suites).

---

## 7. Getting a JoSIM binary

Only needed for §6.4 and the `josim-cli` regression checks. Prebuilt binaries and
source are available from the **[JoSIM project](https://github.com/JoeyDelp/JoSIM)**.
Once installed, point the tests at it:

```bash
JOSIM=/path/to/josim-cli python3 josim_tools/tests/run_tests.py
```

---

## 8. Running the validation suites

| Command | Needs JoSIM? | What it checks |
|:--|:--:|:--|
| `python validation/test_physics.py` | no | bridge physics vs an independent solver & analytic limits (**7/7**) |
| `python validation/run_all.py` | no | reproduces the validation table |
| `JOSIM=… python josim_tools/tests/run_tests.py` | yes | the classical mirror vs the real `josim-cli` (**5/5** on v2.7) |

The two numpy suites are what CI runs on every push (see `.github/workflows/ci.yml`).

---

## 9. Troubleshooting

<details>
<summary><b><code>ModuleNotFoundError: No module named 'josim_bridge'</code></b></summary>

Run from the repository root, or add it to the path. The validation scripts already
do this; for the tools in `josim_tools/`, run them from inside that folder (they add
the parent automatically).
</details>

<details>
<summary><b><code>ModuleNotFoundError: No module named 'numpy'</code></b></summary>

`pip install -r requirements.txt` (or `pip install numpy`). That is the only
dependency.
</details>

<details>
<summary><b>The <code>josim-cli</code> tests are skipped or error</b></summary>

They need a JoSIM binary on `PATH` or via the `JOSIM=` environment variable — see
[§7](#7-getting-a-josim-binary). Everything else runs without it.
</details>

<details>
<summary><b>My anharmonicity doesn't match a paper</b></summary>

Check the operating point. `α ≈ −E_C` only in the deep transmon limit; at finite
`E_J/E_C` there is a `1/√(E_J/E_C)` correction (the validation suite checks this
explicitly). The bridge reports the corrected value.
</details>

---

## 10. Where to go next

- [**`docs/THEORY.md`**](THEORY.md) — the full CPR → potential → Hamiltonian → spectrum derivation, including the `c₄/c₂ = τ/16 − 1/12` result and the references.
- [**`docs/ARCHITECTURE.md`**](ARCHITECTURE.md) — why the quantum solver lives *outside* JoSIM, and how the pieces fit.
- [**`josim_tools/PATCH_SKELETON.md`**](../josim_tools/PATCH_SKELETON.md) — the scoped C++ design for a native `cprtype` selector in JoSIM.
