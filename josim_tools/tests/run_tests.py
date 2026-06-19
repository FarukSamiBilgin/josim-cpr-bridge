"""
run_tests.py  —  regression tests for JoSIM's JJ CPR against known limits.

Drives a JoSIM v2.7 binary, traces I(phi) for each model in exotic_junctions.lib
(plus built-in limits), and checks each against its analytic expectation.

Usage:   JOSIM=/path/to/josim-cli python3 run_tests.py
         (defaults to 'josim-cli' on PATH)

The CPR is traced by ramping the junction phase 0->2pi quasi-statically with a
phase source and reading the device supercurrent (cap=1f, large R, slow ramp).
"""
import os, subprocess, tempfile, sys
import numpy as np

JOSIM = os.environ.get("JOSIM", "josim-cli")
HERE = os.path.dirname(os.path.abspath(__file__))

def trace_cpr(model_defs, model_name):
    """Return (phi, I/Ic) traced from JoSIM for the named model."""
    deck = f"""* CPR trace
{model_defs}
B1 1 0 {model_name} ic=1m
P1 1 0 pwl(0 0 1u 6.283185307)
.tran 0.1n 1u
.print I(B1)
.print P(B1)
.end
"""
    with tempfile.TemporaryDirectory() as d:
        cir = os.path.join(d, "t.cir"); out = os.path.join(d, "t.csv")
        open(cir, "w").write(deck)
        r = subprocess.run([JOSIM, "-m", "-o", out, cir], capture_output=True, text=True)
        if not os.path.exists(out):
            raise RuntimeError("JoSIM did not produce output:\n" + r.stdout + r.stderr)
        data = np.genfromtxt(out, delimiter=",", names=True)
    return data["PB1"], data["IB1"]/1e-3

def transp(phi, tau):
    return np.sin(phi)/np.sqrt(1 - tau*np.sin(phi/2)**2)

def peak_phase(phi, I):
    return phi[np.argmax(I)]

def shape_dev(phi, I, ref):
    g = np.linspace(0.05, 2*np.pi-0.05, 1500)
    Ig = np.interp(g, phi, I, period=2*np.pi); Ig /= np.max(np.abs(Ig))
    Rg = ref(g); Rg /= np.max(np.abs(Rg))
    return np.max(np.abs(Ig-Rg))

results = []
def check(name, cond, detail):
    results.append((name, cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

lib = open(os.path.join(HERE, "..", "exotic_junctions.lib")).read()

print("JoSIM CPR regression tests")
print("="*60)

# 1) D=0 -> ideal sin
defs = ".model sinjj jj(rtype=1 cap=1f r0=1e6 rn=1e6 ic=1m D=0)"
phi, I = trace_cpr(defs, "sinjj")
check("D=0 reduces to sin(phi)", shape_dev(phi, I, np.sin) < 1e-3,
      f"max shape dev = {shape_dev(phi, I, np.sin):.2e}")

# 2) low-T D=0.8 -> exact transparency tau=0.8
defs = ".model d80 jj(rtype=1 cap=1f r0=1e6 rn=1e6 ic=1m D=0.8 T=0.05 TC=9.1)"
phi, I = trace_cpr(defs, "d80")
dev = shape_dev(phi, I, lambda g: transp(g, 0.8))
check("D=0.8 @ T->0 equals transparency tau=0.8", dev < 1e-2, f"max shape dev = {dev:.2e}")

# 3) harmonic library model graphene_tau80 -> transparency tau=0.8 shape
phi, I = trace_cpr(lib, "graphene_tau80")
dev = shape_dev(phi, I, lambda g: transp(g, 0.8))
check("graphene_tau80 (cpr={}) matches tau=0.8", dev < 1.5e-2, f"max shape dev = {dev:.2e}")

# 4) pi-junction -> CPR = -sin(phi) (peak shifted to 3pi/2)
phi, I = trace_cpr(lib, "pi_junction")
pk = peak_phase(phi, I)
check("pi_junction is -sin(phi)", abs(pk - 3*np.pi/2) < 0.05,
      f"peak at phi={pk:.3f} (expect 3pi/2={3*np.pi/2:.3f})")

# 5) forward skew: high-tau peak beyond pi/2
phi, I = trace_cpr(lib, "graphene_tau95")
pk = peak_phase(phi, I)
check("graphene_tau95 is forward-skewed", pk > np.pi/2 + 0.1,
      f"peak at phi={pk:.3f} > pi/2={np.pi/2:.3f}")

print("="*60)
npass = sum(c for _, c in results)
print(f"{npass}/{len(results)} tests passed")
sys.exit(0 if npass == len(results) else 1)
