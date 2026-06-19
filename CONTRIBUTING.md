# Contributing

Thanks for being here. Whether you're fixing a typo, validating against a measured
CPR, or proposing the native JoSIM patch — you're welcome, and this guide will get you
productive fast.

> **The one rule that matters.** This project keeps a *classical* description (JoSIM's
> supercurrent) and a *quantum* description (the bridge's potential) in lock‑step. Any
> change that touches the CPR on one side must keep the physics tests passing on the
> other. Run `python validation/test_physics.py` before you open a PR — green means the
> two ends still agree.

---

## Ways to contribute

- **Validate against real data** — digitize a measured CPR (e.g. ballistic graphene),
  fit it with `cpr_to_josim.py`, and compare the predicted skew/anharmonicity.
- **Extend the tooling** — new junction families in `exotic_junctions.lib`, new gate
  maps, better fitting.
- **The native JoSIM path** — help turn [`PATCH_SKELETON.md`](josim_tools/PATCH_SKELETON.md)
  into a real `cprtype` selector (see the design constraints there — additive, no
  Jacobian, no refactorization changes).
- **Docs, examples, figures** — clarity is a feature.

---

## Development setup

```bash
git clone https://github.com/FarukSamiBilgin/josim-cpr-bridge.git
cd josim-cpr-bridge
pip install -r requirements.txt        # numpy only

python validation/test_physics.py      # 7/7 should pass
```

A JoSIM binary is only needed for the `josim-cli` regression checks — see
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) §7.

---

## The contribution flow

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#0F1828','primaryTextColor':'#EAF0FB','primaryBorderColor':'#2DE0C5','lineColor':'#9DB0D0','secondaryColor':'#121C30','fontFamily':'ui-monospace, monospace'}}}%%
flowchart LR
    F["Fork &<br/>branch"] --> C["Commit<br/>(Conventional)"]
    C --> T["Run<br/>test_physics.py"]
    T -->|green| PR["Open PR"]
    T -->|red| C
    PR --> CI["CI runs<br/>physics checks"]
    CI --> R["Review"]
    R --> M["Merge 🎉"]

    classDef k fill:#0F1828,stroke:#2DE0C5,color:#EAF0FB;
    class F,C,T,PR,CI,R,M k;
```

1. Fork, then branch from `main` with a descriptive name (`feat/graphene-gate-sweep`).
2. Make focused commits using the convention below.
3. Run the physics suite locally — keep it green.
4. Open a PR; fill in the template. CI re‑runs the checks; a maintainer reviews.

---

## Commit style — Conventional Commits

A clean history reads like a changelog. Use
[`<type>(<scope>): <imperative summary>`](https://www.conventionalcommits.org):

| Type | Use for | Example |
|:--|:--|:--|
| `feat` | a new capability | `feat(bridge): add sextic term to taylor_coeffs` |
| `fix` | a bug fix | `fix(cpr): correct sign of c3 under flux bias` |
| `docs` | documentation only | `docs(theory): derive c4/c2 = τ/16 − 1/12` |
| `test` | tests / validation | `test: add π-shift invariance check` |
| `refactor` | no behaviour change | `refactor(spectrum): vectorize charge-basis build` |
| `perf` | performance | `perf(fit): cache the harmonic projection` |
| `build` / `ci` | tooling / pipelines | `ci: run physics checks on 3.8–3.12` |
| `chore` | housekeeping | `chore: update .gitignore` |

Keep the summary in the **imperative mood** ("add", not "added"), under ~72 characters,
and put the *why* in the body when it isn't obvious.

---

## Pull‑request checklist

- [ ] `python validation/test_physics.py` passes (7/7).
- [ ] If the classical mirror changed, the `josim-cli` regression still matches (or you
      explain why it can't run in CI).
- [ ] New behaviour has a check in `validation/`.
- [ ] Docs/figures updated if the public behaviour changed.
- [ ] Commits follow the convention above.

---

## Code style

- **NumPy only.** The core deliberately avoids SciPy/Matplotlib so it installs anywhere
  and runs in CI in seconds. Please keep new core code dependency‑free; put plotting or
  heavy deps in clearly optional scripts.
- Prefer small, pure functions that are easy to test against an analytic limit.
- Match the surrounding style; readable beats clever.

---

## Reporting bugs & requesting features

Use the issue forms (the **New issue** button) — a [bug report](.github/ISSUE_TEMPLATE/bug_report.yml)
or a [feature request](.github/ISSUE_TEMPLATE/feature_request.yml). The more of the form
you fill in (especially a minimal reproducer), the faster it gets resolved.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
