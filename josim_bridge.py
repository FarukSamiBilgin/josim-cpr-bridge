"""
josim_bridge.py  —  JoSIM CPR  ->  qubit Hamiltonian spectrum  (the "bridge")

One model, two solvers. The SAME junction definition that JoSIM uses for
classical time-domain dynamics is here pushed through

    CPR  ->  U(phi) = (hbar/2e) int I(phi') dphi'  ->  H = 4 E_C n^2 + U(phi)
         ->  diagonalize in the charge basis  ->  omega_01, anharmonicity, Kerr.

The CPR is specified in JoSIM's *own* model-card language so the two sides are
provably the same element:
  * cpr   : harmonic amplitude list, I = Ic * sum_n cpr[n] sin(n phi)   (tDep=false branch)
  * D      : point-contact transparency (Haberkorn);   I ~ sin/sqrt(1 - D sin^2(phi/2)) * tanh(...)
  * phiOff : phi_0 offset (pi-junction / anomalous), subtracted inside the sines
This mirrors src/Simulation.cpp (lines ~355-390) of JoSIM v2.7.

Quantum tools (scQubits/SQcircuit) assume a *cosine* well; this module is the
missing piece that accepts an arbitrary JoSIM CPR as a first-class input.
"""
import numpy as np

# BCS-like gap used by JoSIM when tDep is active (mirrors JJ.cpp del0_/del_)
KB = 1.380649e-23
def _delta(T, Tc):
    del0 = 1.76 * KB * Tc
    return del0 * np.sqrt(np.cos((np.pi/2) * (T/Tc)**2))

def cpr_current(phi, cpr=(1.0,), D=0.0, phiOff=0.0, T=None, Tc=9.1):
    """I(phi)/Ic in JoSIM's convention. If D>0 or T given -> Haberkorn branch."""
    cpr = np.atleast_1d(cpr).astype(float)
    ph = phi - phiOff
    sin_phi = sum(cpr[n]*np.sin((n+1)*ph) for n in range(len(cpr)))
    tDep = (D > 0.0) or (T is not None)
    if not tDep:
        return sin_phi                      # pure harmonic CPR (shape; Ic factored out)
    # Haberkorn branch (shape, prefactor/normalization dropped -> peak-normalized later)
    sh = sum(cpr[n]*np.sin((n+1)*ph/2) for n in range(len(cpr)))
    sqrt_part = np.sqrt(1 - D*sh**2)
    if T is None or T == 0:
        return sin_phi/sqrt_part
    d = _delta(T, Tc)
    return (sin_phi/sqrt_part) * np.tanh(d/(2*KB*T) * sqrt_part)

def fundamental(**cpr_kw):
    """First sine-harmonic amplitude b1 of the CPR shape (the natural EJ anchor)."""
    g = np.linspace(0, 2*np.pi, 8192, endpoint=False)
    I = cpr_current(g, **cpr_kw)
    return (2.0/len(g))*np.sum(I*np.sin(g))

def potential_U(phi, EJ=1.0, anchor='fundamental', **cpr_kw):
    """U(phi) = EJ * int_0^phi I(phi')/Ic dphi'  (numeric; EJ = hbar Ic / 2e).

    anchor='fundamental' rescales the CPR so its first harmonic b1=1, so that
    different CPR *shapes* are compared at the same physical well depth (EJ).
    anchor=None keeps JoSIM's raw scaling (fundamental varies with shape)."""
    grid = np.linspace(0, 2*np.pi, 20001)
    I = cpr_current(grid, **cpr_kw)
    if anchor == 'fundamental':
        I = I/fundamental(**cpr_kw)
    Ucum = np.concatenate([[0.0], np.cumsum((I[1:]+I[:-1])/2*np.diff(grid))])
    return EJ*np.interp(np.mod(phi, 2*np.pi), grid, Ucum)

def spectrum(EC, EJ, N=40, ng=0.0, nlev=4, **cpr_kw):
    """Diagonalize H = 4 EC (n-ng)^2 + U(phi) in the charge basis. Returns levels."""
    Ngrid = 4096
    phi = np.linspace(0, 2*np.pi, Ngrid, endpoint=False)
    U = potential_U(phi, EJ=EJ, **cpr_kw)
    uk = np.fft.fft(U)/Ngrid                     # Fourier coeffs: <m|U|n> = u_{m-n}
    ns = np.arange(-N, N+1); dim = 2*N+1
    H = np.zeros((dim, dim), complex)
    H[np.arange(dim), np.arange(dim)] = 4*EC*(ns-ng)**2
    for m in range(dim):
        for n in range(dim):
            H[m, n] += uk[(ns[m]-ns[n]) % Ngrid]
    H = 0.5*(H+H.conj().T)
    return np.sort(np.linalg.eigvalsh(H).real)[:nlev]

def spectrum_vecs(EC, EJ, N=40, ng=0.0, nlev=4, **cpr_kw):
    """Like spectrum() but also returns eigenvectors and the charge grid ns."""
    Ngrid = 4096
    phi = np.linspace(0, 2*np.pi, Ngrid, endpoint=False)
    U = potential_U(phi, EJ=EJ, **cpr_kw)
    uk = np.fft.fft(U)/Ngrid
    ns = np.arange(-N, N+1); dim = 2*N+1
    H = np.zeros((dim, dim), complex)
    H[np.arange(dim), np.arange(dim)] = 4*EC*(ns-ng)**2
    for m in range(dim):
        for n in range(dim):
            H[m, n] += uk[(ns[m]-ns[n]) % Ngrid]
    H = 0.5*(H+H.conj().T)
    w, v = np.linalg.eigh(H)
    idx = np.argsort(w.real)
    return w.real[idx][:nlev], v[:, idx][:, :nlev], ns

def qubit_params(EC, EJ, **cpr_kw):
    """Convenience: omega_01, anharmonicity alpha (=Kerr), in same units as EC/EJ."""
    e = spectrum(EC, EJ, **cpr_kw)
    w01 = e[1]-e[0]; w12 = e[2]-e[1]
    return w01, (w12-w01)

def charge_mat_elem(EC, EJ, **cpr_kw):
    """|<0|n|1>|^2  (drives charge/dielectric relaxation; n is diagonal in charge basis)."""
    w, v, ns = spectrum_vecs(EC, EJ, **cpr_kw)
    n01 = np.vdot(v[:, 0], ns*v[:, 1])
    return abs(n01)**2

# physical constants (SI)
_E = 1.602176634e-19; _HBAR = 1.054571817e-34; _KB = 1.380649e-23; _H = 6.62607015e-34
def t1_estimate(EC, EJ, tan_delta=1e-6, T_bath=0.02, **cpr_kw):
    """T1 from dielectric (capacitive) loss, the dominant transmon channel.
        1/T1 = |<0|n|1>|^2 * (8 E_C / hbar) * tan_delta * [1 + coth(hbar w01 / 2 kT)]/2
    Standard loss-tangent model (cf. scQubits t1_capacitive). EC,EJ in GHz; tan_delta
    is the dielectric loss tangent (typical 1e-6). RATIO vs cosine is model-robust;
    absolute value scales linearly with tan_delta."""
    w01 = 2*np.pi*qubit_params(EC, EJ, **cpr_kw)[0]*1e9      # rad/s
    n01sq = charge_mat_elem(EC, EJ, **cpr_kw)
    EC_J = EC*1e9*_H
    therm = 0.5*(1.0 + 1.0/np.tanh(_HBAR*w01/(2*_KB*T_bath)))
    Gamma1 = n01sq * (8*EC_J/_HBAR) * tan_delta * therm
    return 1.0/Gamma1                                        # seconds

def taylor_coeffs(EJ=1.0, **cpr_kw):
    """c2, c3, c4 of U about its minimum: c2->freq, c3->three-wave g3, c4->Kerr."""
    phi = np.linspace(-0.6, 0.6, 4001)
    # locate minimum near 0
    g = np.linspace(-np.pi, np.pi, 20001); Ug = potential_U(g, EJ=EJ, **cpr_kw)
    phi0 = g[np.argmin(Ug)]
    x = phi; U = potential_U(phi0+x, EJ=EJ, **cpr_kw)
    p = np.polyfit(x, U, 6)               # U ~ sum p_k x^k
    return {'c2': p[-3], 'c3': p[-4], 'c4': p[-5], 'phi_min': phi0}

if __name__ == "__main__":
    EC, EJ = 0.25, 12.5      # GHz ; E_J/E_C = 50, transmon regime
    
    print("=== full bridge output: spectrum + Kerr + T1 ===")
    def report(label, **kw):
        w01, a = qubit_params(EC, EJ, **kw)
        n01 = charge_mat_elem(EC, EJ, **kw)
        t1 = t1_estimate(EC, EJ, **kw)
        print(f"{label:24s} w01={w01:6.3f} GHz  alpha={a*1000:7.1f} MHz  "
              f"|<0|n|1>|^2={n01:6.4f}  T1={t1*1e6:7.2f} us")
        return t1
    t1_cos = report("cosine (cpr={1})", cpr=(1.0,))
    for tau in [0.3, 0.6, 0.9]:
        t1 = report(f"tau={tau} (D, T=0)", D=tau, T=0)
        print(f"{'':24s}  -> T1 / T1_cosine = {t1/t1_cos:.3f}")
