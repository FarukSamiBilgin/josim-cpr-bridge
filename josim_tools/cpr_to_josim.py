"""
cpr_to_josim.py  —  fit any current-phase relation (CPR) to a JoSIM cpr={...} card.

Input: a CPR as samples I(phi) over [0,2pi) (from experiment / DFT / a model),
or a callable.  Output: the c1-normalized harmonic vector that JoSIM consumes as
  .model name jj(... cpr={c1, c2, ...})   with   I = Ic * sum_n cn sin(n phi),
plus the harmonic count needed for a target shape accuracy and the residual.

Mirrors JoSIM v2.7's harmonic CPR branch (src/Simulation.cpp).  Pure numpy.
"""
import numpy as np

def sine_coeffs(phi, I, nmax=16):
    """Fourier sine coefficients b_n of I(phi) (assumed odd, 2pi-periodic)."""
    g = np.linspace(0, 2*np.pi, 8192, endpoint=False)
    Ig = np.interp(g, np.mod(phi, 2*np.pi), I, period=2*np.pi)
    return np.array([(2.0/len(g))*np.sum(Ig*np.sin(n*g)) for n in range(1, nmax+1)])

def _resid(phi, I, b):
    g = np.linspace(0, 2*np.pi, 8192, endpoint=False)
    Ig = np.interp(g, np.mod(phi, 2*np.pi), I, period=2*np.pi)
    ap = sum(b[n-1]*np.sin(n*g) for n in range(1, len(b)+1))
    return np.sqrt(np.mean((Ig-ap)**2))/np.max(np.abs(Ig))

def fit_cpr(phi, I, target=1e-2, nmax=16):
    """Return dict: cpr vector (c1-normalized), n_harmonics, residual, model line."""
    b_full = sine_coeffs(phi, I, nmax)
    n = nmax
    for k in range(1, nmax+1):
        if _resid(phi, I, b_full[:k]) < target:
            n = k; break
    b = b_full[:n]/b_full[0]
    return {
        'cpr': b,
        'n_harmonics': n,
        'residual': _resid(phi, I, b_full[:n]),
        'card': "cpr={" + ", ".join(f"{x:.4f}" for x in b) + "}",
    }

def transparency_cpr(phi, tau):
    """Short-ballistic / Beenakker transparency CPR (graphene/SNS/Dirac)."""
    return np.sin(phi)/np.sqrt(1 - tau*np.sin(phi/2)**2)

def model_line(name, fit, ic="1m", cap="50f", rn="16", r0="160"):
    """Assemble a full JoSIM .model card from a fit result."""
    return f".model {name} jj(rtype=1 ic={ic} cap={cap} rn={rn} r0={r0} {fit['card']})"

if __name__ == "__main__":
    phi = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    print("Fitting transparency CPRs to JoSIM cpr={...} (target 1% shape):")
    for tau in [0.6, 0.8, 0.95]:
        f = fit_cpr(phi, transparency_cpr(phi, tau), target=1e-2)
        print(f"  tau={tau}: {f['n_harmonics']} harmonics, resid={f['residual']:.2e}")
        print("     ", model_line(f"graphene_tau{int(tau*100)}", f))
