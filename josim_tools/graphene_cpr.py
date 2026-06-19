"""
graphene_cpr.py  —  reference implementation for a gate-tunable graphene/Dirac CPR.

This is the *reference behaviour* for the proposed JoSIM `cprtype=graphene` model:
given a (static) gate voltage Vg, it produces the short-ballistic transparency CPR
    I(phi) = Ic(Vg) * sin(phi) / sqrt(1 - tau(Vg) * sin^2(phi/2))
and emits the equivalent JoSIM model card (via the existing cpr={...} path, or D).

The C++ implementation in Simulation::handle_jj should reproduce these numbers.
The gate mapping tau(Vg), Ic(Vg) is PHENOMENOLOGICAL and calibratable to data;
defaults capture the measured behaviour of ballistic graphene JJs:
  - skewness (set by tau) increases as the gate moves away from the Dirac point,
  - critical current grows with carrier density (~sqrt for graphene DOS),
  - an optional Fabry-Perot term reproduces the gate oscillation of skewness
    reported by Nanda et al., Nano Lett. 17, 3396 (2017) (arXiv:1612.06895).
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpr_to_josim import fit_cpr, transparency_cpr, model_line

def tau_of_vg(Vg, tau_max=0.97, V_dirac=0.0, V0=1.0, fp_amp=0.0, V_fp=1.5):
    """Gate-tunable channel transparency (phenomenological, calibratable).
    Rises from ~0 at the Dirac point toward tau_max away from it; optional
    Fabry-Perot oscillation (fp_amp>0) for the gate-modulated skewness."""
    x = abs(Vg - V_dirac)
    base = tau_max*(1.0 - np.exp(-x/V0))
    fp = fp_amp*np.cos(2*np.pi*Vg/V_fp)
    return float(np.clip(base + fp, 1e-3, 0.999))

def ic_of_vg(Vg, Ic0=1e-3, V_dirac=0.0, n0=1.0):
    """Gate-tunable critical current (~graphene DOS, sqrt of carrier density)."""
    return float(Ic0*np.sqrt(abs(Vg - V_dirac)/n0 + 1e-3))

def graphene_cpr(phi, Vg, **gate_kw):
    tau = tau_of_vg(Vg, **{k: v for k, v in gate_kw.items()
                           if k in ('tau_max','V_dirac','V0','fp_amp','V_fp')})
    return transparency_cpr(phi, tau)

def graphene_model_card(name, Vg, target=1e-2, **gate_kw):
    """JoSIM .model card for a graphene JJ at gate voltage Vg (harmonic route)."""
    phi = np.linspace(0, 2*np.pi, 2000, endpoint=False)
    tau = tau_of_vg(Vg, **{k: v for k, v in gate_kw.items()
                           if k in ('tau_max','V_dirac','V0','fp_amp','V_fp')})
    ic = ic_of_vg(Vg, **{k: v for k, v in gate_kw.items() if k in ('Ic0','V_dirac','n0')})
    f = fit_cpr(phi, transparency_cpr(phi, tau), target=target)
    return tau, ic, model_line(name, f, ic=f"{ic:.4e}"), f['n_harmonics']

if __name__ == "__main__":
    print("Gate-tunable graphene CPR reference (static Vg):")
    print(f"{'Vg':>6}{'tau':>8}{'Ic[mA]':>10}{'#harm':>7}   model card")
    for Vg in [0.2, 0.5, 1.0, 2.0, 4.0]:
        tau, ic, card, nh = graphene_model_card(f"g_vg{int(Vg*10)}", Vg)
        print(f"{Vg:6.1f}{tau:8.3f}{ic*1e3:10.3f}{nh:7d}   {card}")
