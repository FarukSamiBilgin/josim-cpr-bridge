"""Reproduce the validation table and the bridge demonstration figure.
Pure Python (numpy/scipy/matplotlib); does not require josim-cli.
The JoSIM-side checks (D=0 -> sin, CPR={...} through JoSIM) use
../josim_decks/cpr_tracer.cir with a JoSIM v2.7 binary."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import josim_bridge as jb

EC, EJ = 0.25, 12.5

def harmvec(tau, N):
    g = np.linspace(0, 2*np.pi, 8192, endpoint=False)
    f = np.sin(g)/np.sqrt(1 - tau*np.sin(g/2)**2)
    b = [(2/len(g))*np.sum(f*np.sin(n*g)) for n in range(1, N+1)]
    return tuple(np.array(b)/b[0])

print("== cosine baseline ==")
w, a = jb.qubit_params(EC, EJ, cpr=(1.0,))
print(f"  omega01 = {w:.4f} GHz   alpha = {a*1000:.1f} MHz   (expect alpha ~ -E_C)")

print("== harmonic route vs exact-D route (tau=0.9) ==")
wh, ah = jb.qubit_params(EC, EJ, cpr=harmvec(0.9, 8))
wd, ad = jb.qubit_params(EC, EJ, D=0.9, T=0)
print(f"  omega01: harmonic(8)={wh:.4f}  exact-D={wd:.4f}  (diff {abs(wh-wd)/wd*100:.2f}%)")
print(f"  alpha  : harmonic(8)={ah*1000:.2f}  exact-D={ad*1000:.2f} MHz")

print("== anharmonicity convergence (tau=0.9) ==")
exact = jb.qubit_params(EC, EJ, D=0.9, T=0)[1]*1000
for N in [2, 3, 5, 8, 12]:
    aN = jb.qubit_params(EC, EJ, cpr=harmvec(0.9, N))[1]*1000
    print(f"  {N:2d} harmonics: alpha = {aN:7.2f} MHz   (exact-D {exact:.2f})")

print("== full output incl. T1 (dielectric, tan_delta=1e-6) ==")
t1c = jb.t1_estimate(EC, EJ, cpr=(1.0,))
for tau in [0.0, 0.3, 0.6, 0.9]:
    kw = dict(cpr=(1.0,)) if tau == 0 else dict(D=tau, T=0)
    w, a = jb.qubit_params(EC, EJ, **kw)
    t1 = jb.t1_estimate(EC, EJ, **kw)
    print(f"  tau={tau:.1f}: w01={w:.3f} GHz  alpha={a*1000:7.1f} MHz  "
          f"T1={t1*1e6:6.1f} us  (T1/T1_cos={t1/t1c:.3f})")

print("\nValidation complete.")
