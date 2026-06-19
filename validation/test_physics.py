"""
test_physics.py  —  physics validation of the bridge (pure numpy; no josim-cli).

These guard against "runs fine but wrong physics": each check compares the bridge
against an INDEPENDENT computation or a known analytic result.
Run:  python3 validation/test_physics.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import josim_bridge as jb

fails = []
def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond: fails.append(name)

# Independent solver: phase-basis finite difference on a ring (vs charge basis)
def spectrum_phasebasis(EC, EJ, Ngrid=2001, nlev=3, **cpr_kw):
    phi = np.linspace(-np.pi, np.pi, Ngrid, endpoint=False); d = phi[1]-phi[0]
    U = jb.potential_U(phi+np.pi, EJ=EJ, **cpr_kw)
    main = np.full(Ngrid, -2.0); off = np.ones(Ngrid-1)
    T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    T[0, -1] = 1; T[-1, 0] = 1
    H = -4*EC*T/d**2 + np.diag(U)
    return np.sort(np.linalg.eigvalsh(H))[:nlev]

EC, EJ = 0.25, 12.5

# 1) two independent solvers must agree (charge basis vs phase-basis FD)
for label, kw in [("cosine", dict(cpr=(1.0,))), ("tau=0.9", dict(D=0.9, T=0))]:
    e = jb.spectrum(EC, EJ, **kw); w_c = e[1]-e[0]; a_c = (e[2]-e[1])-(e[1]-e[0])
    p = spectrum_phasebasis(EC, EJ, **kw); w_p = p[1]-p[0]; a_p = (p[2]-p[1])-(p[1]-p[0])
    check(f"independent solvers agree ({label})",
          abs(w_c-w_p) < 5e-3 and abs(a_c-a_p) < 2e-3,
          f"w01 {w_c:.4f}/{w_p:.4f} GHz, alpha {a_c*1e3:.1f}/{a_p*1e3:.1f} MHz")

# 2) cosine omega01 vs analytic sqrt(8 EJ EC) - EC  (Koch 2007)
w01 = jb.qubit_params(EC, EJ, cpr=(1.0,))[0]; wa = np.sqrt(8*EJ*EC)-EC
check("cosine omega01 vs sqrt(8EjEc)-Ec", abs(w01-wa)/wa < 0.03, f"{w01:.4f} vs {wa:.4f} GHz")

# 3) alpha/EC -> -1 with the known ~1/sqrt(EJ/EC) correction (not a fixed tolerance!)
devs = [jb.qubit_params(1.0, float(r), cpr=(1.0,), N=60)[1]+1 for r in [100, 400, 1600]]
monotone = abs(devs[0]) > abs(devs[1]) > abs(devs[2])
scaling = abs(devs[2]) < 0.04           # 1/sqrt(1600)=0.025 -> deviation should be small
check("alpha/EC -> -1 as EJ/EC grows (1/sqrt law)", monotone and scaling,
      f"deviations {[round(x,4) for x in devs]} at EJ/EC=100,400,1600")

# 4) Taylor c4/c2 of transparency matches symbolic law tau/16 - 1/12
errs = []
for tau in [0.5, 0.9]:
    tc = jb.taylor_coeffs(EJ=1.0, D=tau, T=0)
    errs.append(abs(tc['c4']/tc['c2'] - (tau/16-1/12)))
check("c4/c2 matches symbolic tau/16-1/12", max(errs) < 3e-3, f"max err {max(errs):.2e}")

# 5) pi-shift (PHI) leaves the spectrum invariant
e0 = jb.spectrum(EC, EJ, cpr=(1.0,)); ep = jb.spectrum(EC, EJ, cpr=(1.0,), phiOff=np.pi)
check("pi-shift invariant spectrum", np.allclose(e0-e0[0], ep-ep[0], atol=1e-6),
      f"max diff {np.max(np.abs((e0-e0[0])-(ep-ep[0]))):.1e}")

# 6) D->0 reduces exactly to cosine
ec = jb.qubit_params(EC, EJ, cpr=(1.0,)); ed = jb.qubit_params(EC, EJ, D=1e-6, T=0)
check("D->0 equals cosine", abs(ec[0]-ed[0]) < 1e-4 and abs(ec[1]-ed[1]) < 1e-4,
      f"w01 diff {abs(ec[0]-ed[0])*1e3:.3f} MHz")

print(f"\n{len(fails)==0 and 'ALL PHYSICS CHECKS PASS' or 'FAILURES: '+str(fails)}")
sys.exit(0 if not fails else 1)
