# `validation/` — proof the physics is right

Two independent layers guard against "runs fine but wrong physics." Each check compares
the bridge against an *independent* computation or a known analytic result.

| File | Needs JoSIM? | What it proves |
|:--|:--:|:--|
| [`test_physics.py`](test_physics.py) | no | The bridge vs an **independent** eigensolver (phase‑basis finite difference) and analytic limits — including `c₄/c₂ = τ/16 − 1/12` and π‑shift invariance. **7/7.** |
| [`run_all.py`](run_all.py) | no | Reproduces the validation table end to end. |

```bash
python validation/test_physics.py     # 7/7 — this is what CI runs on every push
python validation/run_all.py
```

The `josim-cli` regression (the classical mirror vs the real binary, 5/5 on v2.7) lives
in [`../josim_tools/tests/run_tests.py`](../josim_tools/tests/run_tests.py).
