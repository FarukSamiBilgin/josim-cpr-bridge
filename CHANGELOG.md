# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Validation against a published, measured ballistic-graphene CPR (fit `τ(V_g)`, reproduce the skewness-vs-gate trend).
- Native `cprtype` selector in JoSIM (`harmonic` / `graphene` / `table`) — see [`PATCH_SKELETON.md`](josim_tools/PATCH_SKELETON.md).

## [0.1.0] — 2026-06-18

### Added
- **The bridge** (`josim_bridge.py`): `cpr_current`, `potential_U`, `spectrum`, `qubit_params`, `taylor_coeffs`, `charge_mat_elem`, `t1_estimate` — CPR → U(φ) → H → spectrum, in pure NumPy.
- **JoSIM-side tooling** (`josim_tools/`): `cpr_to_josim.py` (fit any CPR to a `cpr={…}` card), `graphene_cpr.py` (gate-tunable reference `τ(V_g)`, `I_c(V_g)`), `exotic_junctions.lib` (graphene / SNS / Dirac, π- and φ₀-junctions), and `PATCH_SKELETON.md` (scoped C++ design for native CPR).
- **JoSIM deck** (`josim_decks/cpr_tracer.cir`): traces `I(φ)` for sin / transparency / harmonic CPRs.
- **Validation**: `validation/test_physics.py` (7/7 — independent solver + analytic limits) and `validation/run_all.py`; `josim_tools/tests/run_tests.py` (5/5 regression on the JoSIM v2.7 binary).
- **Documentation**: a visual README, a 0-to-100 getting-started guide, a theory note, and an architecture document, with custom SVG banner and diagrams.

### Result
- Established that the cosine-locked ratio `c₄/c₂ = −1/12` becomes `τ/16 − 1/12` for a transparency CPR — making anharmonicity a **design knob** (~3× tunable at fixed `E_C`), while `T₁` improves modestly with transparency.
- Confirmed (on the JoSIM v2.7 binary) that the temperature-dependent Haberkorn branch already reduces to the Beenakker transparency CPR at `T → 0`, so the canonical graphene CPR is representable in stock JoSIM.

[Unreleased]: https://github.com/FarukSamiBilgin/josim-cpr-bridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FarukSamiBilgin/josim-cpr-bridge/releases/tag/v0.1.0
