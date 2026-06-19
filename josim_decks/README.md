# `josim_decks/` — runnable JoSIM netlists

SPICE‑style decks you can run with a JoSIM binary. They are how the classical side is
exercised directly, and they generate the data behind the validation tables.

| File | What it does |
|:--|:--|
| [`cpr_tracer.cir`](cpr_tracer.cir) | Quasi‑statically ramps a junction phase `0 → 2π` and reads the supercurrent, so the output traces `I(φ)/I_c`. Compares ideal `sin φ`, JoSIM's built‑in `D=0.9` skew, and a 5‑harmonic fit side by side. |

Run it:

```bash
josim-cli -m -o out.csv cpr_tracer.cir
```

Need a binary? See [`../docs/GETTING_STARTED.md`](../docs/GETTING_STARTED.md) §7.
