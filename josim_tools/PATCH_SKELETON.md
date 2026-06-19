# JoSIM patch skeleton — `cprtype` (graphene / table) CPR

**Status: design skeleton.** Compile/iterate on a machine with the JoSIM build
toolchain. Grounded in JoSIM v2.7 source (`src/Model.cpp`, `include/JoSIM/Model.hpp`,
`src/Simulation.cpp` supercurrent block ~ lines 355–390). Additive and non-breaking:
absence of `cprtype` preserves the existing harmonic default exactly. Per Johannes'
note, the CPR enters via the predicted phase `phi0` on the RHS — **no `dI/dphi`
Jacobian term**, and the voltage-region conductance / refactorization logic is untouched.

## 1. `Model` — new state (`Model.hpp`)

```cpp
enum class CPRType { Harmonic, Graphene, Table };   // Harmonic = current default
CPRType  cprType_   = CPRType::Harmonic;
double   tau_       = 0.0;                 // transparency (graphene); 0 -> sin
std::vector<double> cprPhi_, cprI_;        // tabulated I(phi) (Table), normalized
// accessors mirroring existing ones (cpr(), d(), phiOff(), ...)
```

## 2. `Model.cpp` — parse new tokens (next to the existing `D`, `T`, `CPR` cases)

```cpp
else if (tokens.at(i) == "CPRTYPE") {
    std::string v = tokens.at(i+1);
    if      (v == "graphene") temp.cprType(CPRType::Graphene);
    else if (v == "table")    temp.cprType(CPRType::Table);
    else                      temp.cprType(CPRType::Harmonic);
}
else if (tokens.at(i) == "TAU")     { temp.tau(value); }            // graphene
else if (tokens.at(i) == "CPRFILE") { load_cpr_table(temp, token); } // table -> cprPhi_/cprI_
```
`load_cpr_table` reads a 2-column `phi,I` file, sorts, normalizes to unit peak,
and stores it (so `max|I| = Ic` after the existing `ic`/`area` scaling — see §4).
For **static gate** Vg, `tau`/`ic` are precomputed by the user (or a helper) and
entered as fixed parameters → nothing changes at runtime (no refactorization).

## 3. `Simulation.cpp` — dispatch in the supercurrent evaluation (tDep=false branch)

The current line builds `ic_sin_phi = sum_h cpr[h]*sin((h+1)*(phi0 - phiOff))`.
Replace with a dispatch on `cprType_`, all evaluated at the predicted `phi0`:

```cpp
double cpr_eval(const Model& m, double ph /* = phi0 - phiOff */) {
  switch (m.cprType()) {
    case CPRType::Harmonic: {                       // unchanged path
      double s = 0.0;
      for (size_t h = 0; h < m.cpr().size(); ++h) s += m.cpr()[h]*std::sin((h+1)*ph);
      return s;
    }
    case CPRType::Graphene:                          // transparency form
      return std::sin(ph) / std::sqrt(1.0 - m.tau()*std::pow(std::sin(ph/2.0), 2));
    case CPRType::Table:                             // tabulated, 2pi-periodic interp
      return interp_periodic(m.cprPhi(), m.cprI(), wrap_2pi(ph));
  }
}
// then:  ic_sin_phi = m.ic() * cpr_eval(m, phi0 - m.phiOff());   // RHS only
```

Note: `Graphene` reuses the same math as the existing `D`/Haberkorn branch at T→0,
so it can alternatively delegate to that code path; kept explicit here for clarity.

## 4. Normalization (the open question Johannes raised)

Two consistent choices — pick one and apply uniformly:
- **(A) peak = Ic:** normalize `cpr_eval` shape to unit peak, so `max_phi I = ic*area`.
  Matches the harmonic `cpr={...}` intuition; Table input is rescaled to unit peak on load.
- **(B) Ambegaokar–Baratoff:** carry the `pi*Delta/2eRn` prefactor as the existing
  `D`/Haberkorn branch does, for consistency with that path.

The reference tool (`josim_tools/graphene_cpr.py`) currently assumes (A).

## 5. Tests (extend `josim_tools/tests/run_tests.py`)

- Regression: every existing `cpr={...}` and `D` model traces **bit-identically**
  (cprType defaults to Harmonic).
- New: `cprtype=graphene tau=0.8` matches `cprtype=table` loaded with the transparency
  curve, and both match the `D=0.8 T->0` branch — three routes, one shape.
- Gate sweep: `graphene_cpr.py` reference values reproduced by JoSIM to < 1% (the PoC).

## Files to touch
`include/JoSIM/Model.hpp`, `src/Model.cpp`, `src/Simulation.cpp` (+ a small
`interp_periodic`/`load_cpr_table` helper). No matrix-assembly or refactorization changes.
