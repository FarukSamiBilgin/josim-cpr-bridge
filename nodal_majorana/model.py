"""
Bogoliubov-de Gennes model for vortex-bound Majorana modes on a nodal
two-dimensional superconductor.

Physical setting
----------------
A single-band square-lattice metal with Rashba spin-orbit coupling and an
out-of-plane Zeeman field, proximitized by a spin-singlet pair potential whose
momentum-space form factor is a free choice:

    s          Delta(k) = D0                                   (fully gapped)
    ext-s      Delta(k) = D0 * [ds + (cos kx + cos ky) / 2]     (gapped -> nodal)
    d          Delta(k) = D0 * (cos kx - cos ky) / 2            (nodal, always)

The normal part is the standard Sau-Lutchyn-Tewari-Das Sarma construction,

    h(k) = xi(k) s0 + alpha (sin ky sx - sin kx sy) + Vz sz,
    xi(k) = 2 t (2 - cos kx - cos ky) - mu,

which for an s-wave form factor is a topological superconductor whenever
Vz^2 > D0^2 + mu^2 (band bottom at Gamma).  A vortex then binds one Majorana
zero mode.  Replacing the s-wave form factor by a nodal one keeps the vortex
but opens gapless quasiparticle channels along the nodal directions -- which is
exactly the regime whose hybridization law this module measures.

Nambu convention
----------------
Per site the basis is (c_up, c_dn, c_up^dag, c_dn^dag), and

    H_BdG = [[ h ,  D  ],
             [ D^dag , -h^* ]],        D(k) = Delta(k) * i sy.

Real-space matrix elements are obtained from the k-space form by the usual
correspondence: a term A cos(k_n) contributes A/2 to both (+n) and (-n) bonds,
and a term A sin(k_n) contributes A/(2i) to (+n) and -A/(2i) to (-n).

Vortices
--------
A vortex configuration multiplies every pair-potential matrix element by

    Delta -> Delta * f(r) * exp(i theta(r)),
    exp(i theta(r)) = prod_v (r - R_v) / |r - R_v|,
    f(r)            = prod_v tanh(|r - R_v| / xi0),

evaluated at the bond midpoint (at the site itself for the on-site term).
Writing the phase as a product of unit complex numbers keeps it single valued:
no branch cut is ever crossed, so open boundary conditions are consistent for
any number of vortices.

Only the pair potential carries the vortex; the vector potential is dropped.
This is the standard extreme type-II / London-limit treatment, appropriate when
the magnetic length is far larger than both the coherence length and the vortex
separations scanned here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

# Pauli matrices in spin space.
S0 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# The singlet pairing matrix i*sy.
I_SY = 1j * SY  # [[0, 1], [-1, 0]]

FORM_FACTORS = ("s", "ext-s", "d")


@dataclass(frozen=True)
class Model:
    """Parameters of the lattice BdG model.

    Energies are in units of the hopping ``t``; lengths in lattice constants.

    Attributes
    ----------
    t : nearest-neighbour hopping.
    mu : chemical potential measured from the band bottom at Gamma.
    alpha : Rashba spin-orbit strength.
    vz : out-of-plane Zeeman field.
    d0 : pair-potential amplitude.
    form : pairing form factor, one of ``FORM_FACTORS``.
    ds : offset of the extended-s form factor (ignored otherwise).  Large ``ds``
        is nearly isotropic; the gap develops nodes once the line
        ``ds + (cos kx + cos ky)/2 = 0`` crosses the Fermi surface.
    d_is : amplitude of an additional on-site ``i * d_is`` pairing component.
        Added to a nodal form factor this is the ``d + is`` state: it breaks
        time reversal and opens a gap of size ``d_is`` exactly at the nodes,
        while leaving the antinodal gap essentially untouched.  It is the node
        regulator -- the Majorana length along the nodal direction scales as
        ``1 / d_is``, so sending ``d_is -> 0`` widens the window in which the
        nodal power law would be visible without ever working directly with a
        gapless spectrum.  It also shrinks the core-level minigap at the same
        rate, and the pair is only resolvable inside that minigap, so the two
        effects work against each other; see ``run_exponent.py``.
    xi0 : core-size parameter of the ``tanh`` amplitude profile.
    """

    t: float = 1.0
    mu: float = 1.0
    alpha: float = 1.0
    vz: float = 1.2
    d0: float = 0.5
    form: str = "s"
    ds: float = 1.0
    d_is: float = 0.0
    xi0: float = 2.0

    def __post_init__(self) -> None:
        if self.form not in FORM_FACTORS:
            raise ValueError(f"form must be one of {FORM_FACTORS}, got {self.form!r}")

    # ---------------------------------------------------------------- k space

    def form_factor(self, kx, ky):
        """Momentum-space pairing form factor Delta(k) / 1 (already carries d0)."""
        kx = np.asarray(kx, dtype=float)
        ky = np.asarray(ky, dtype=float)
        if self.form == "s":
            base = self.d0 * np.ones_like(kx)
        elif self.form == "ext-s":
            base = self.d0 * (self.ds + 0.5 * (np.cos(kx) + np.cos(ky)))
        else:
            base = self.d0 * 0.5 * (np.cos(kx) - np.cos(ky))
        return base + 1j * self.d_is if self.d_is else base

    def xi(self, kx, ky):
        """Normal-state dispersion measured from the chemical potential."""
        kx = np.asarray(kx, dtype=float)
        ky = np.asarray(ky, dtype=float)
        return 2.0 * self.t * (2.0 - np.cos(kx) - np.cos(ky)) - self.mu

    def h_k(self, kx: float, ky: float) -> np.ndarray:
        """Normal-state 2x2 Bloch Hamiltonian."""
        return (
            self.xi(kx, ky) * S0
            + self.alpha * (np.sin(ky) * SX - np.sin(kx) * SY)
            + self.vz * SZ
        )

    def bdg_k(self, kx: float, ky: float) -> np.ndarray:
        """Uniform-Delta 4x4 BdG Bloch Hamiltonian (no vortex)."""
        h = self.h_k(kx, ky)
        hm = self.h_k(-kx, -ky)
        d = self.form_factor(kx, ky) * I_SY
        top = np.hstack([h, d])
        bot = np.hstack([d.conj().T, -hm.conj()])
        return np.vstack([top, bot])

    def bulk_gap(self, n: int = 241) -> float:
        """Minimum positive BdG eigenvalue over a Brillouin-zone mesh.

        A nodal form factor drives this to zero as the mesh is refined; a gapped
        one converges to the true spectral gap.
        """
        ks = np.linspace(-np.pi, np.pi, n, endpoint=False)
        best = np.inf
        for kx in ks:
            for ky in ks:
                ev = np.linalg.eigvalsh(self.bdg_k(kx, ky))
                best = min(best, float(np.min(np.abs(ev))))
        return best

    def is_topological_at_gamma(self) -> bool:
        """Sau et al. criterion for the band-bottom band inversion.

        Necessary but *not* sufficient: it only tracks the gap closing at Gamma.
        Use :meth:`chern_number` to decide whether a vortex actually binds a
        Majorana mode.
        """
        return self.vz**2 > abs(self.form_factor(0.0, 0.0)) ** 2 + self.mu**2

    def chern_number(self, n: int = 48) -> int:
        """Chern number of the filled BdG bands (Fukui-Hatsugai lattice method).

        A vortex binds one Majorana zero mode when this is odd.  The method is
        gauge invariant and returns an integer on any mesh fine enough to
        resolve the gap; it is meaningless for a nodal form factor, where the
        filled bands touch the empty ones.
        """
        ks = 2 * np.pi * np.arange(n) / n
        nb = 2  # number of negative-energy BdG bands
        occ = np.empty((n, n, 4, nb), dtype=complex)
        for a, kx in enumerate(ks):
            for b, ky in enumerate(ks):
                w, v = np.linalg.eigh(self.bdg_k(kx, ky))
                occ[a, b] = v[:, :nb]

        def link(u1, u2):
            m = np.linalg.det(u1.conj().T @ u2)
            return m / abs(m) if abs(m) > 0 else 1.0 + 0j

        total = 0.0
        for a in range(n):
            for b in range(n):
                a2, b2 = (a + 1) % n, (b + 1) % n
                u = (
                    link(occ[a, b], occ[a2, b])
                    * link(occ[a2, b], occ[a2, b2])
                    * link(occ[a2, b2], occ[a, b2])
                    * link(occ[a, b2], occ[a, b])
                )
                total += np.angle(u)
        return int(np.rint(total / (2 * np.pi)))

    def fermi_surface(self, angle_deg: float = 0.0, n: int = 20001
                      ) -> list[tuple[float, float]]:
        """Fermi crossings along a ray from Gamma, from the *lattice* bands.

        Returns a list of ``(k_F, v_F)`` pairs -- one per crossing of a
        normal-state band through zero along the ray -- with ``v_F`` the radial
        Fermi velocity ``|d eps / dk|`` obtained by central difference.

        These are independent reference values: the Majorana localization length
        of the gapped benchmark is ``xi_M = v_F / Delta_eff`` and the splitting
        oscillates with period ``2 pi / k_F``, both of which the fits must
        reproduce without ever being told them.
        """
        ang = np.deg2rad(angle_deg)
        ux, uy = np.cos(ang), np.sin(ang)
        ks = np.linspace(0.0, np.pi, n)
        bands = np.array(
            [np.linalg.eigvalsh(self.h_k(k * ux, k * uy)) for k in ks]
        )  # (n, 2)
        out: list[tuple[float, float]] = []
        for b in range(bands.shape[1]):
            e = bands[:, b]
            idx = np.nonzero(np.sign(e[:-1]) != np.sign(e[1:]))[0]
            for i in idx:
                k0, k1 = ks[i], ks[i + 1]
                e0, e1 = e[i], e[i + 1]
                kf = float(k0 - e0 * (k1 - k0) / (e1 - e0))
                dk = 1e-4
                em = np.linalg.eigvalsh(self.h_k((kf - dk) * ux, (kf - dk) * uy))[b]
                ep = np.linalg.eigvalsh(self.h_k((kf + dk) * ux, (kf + dk) * uy))[b]
                out.append((kf, float(abs(ep - em) / (2 * dk))))
        return out


# --------------------------------------------------------------------- vortex


@dataclass(frozen=True)
class VortexConfig:
    """Positions of unit vortices, in lattice coordinates."""

    centers: tuple[tuple[float, float], ...] = field(default=())

    def order_parameter(self, x, y, xi0: float) -> np.ndarray:
        """Complex profile ``f(r) exp(i theta(r))`` at the given points."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        out = np.ones(np.broadcast(x, y).shape, dtype=complex)
        for cx, cy in self.centers:
            dx = x - cx
            dy = y - cy
            r = np.hypot(dx, dy)
            phase = np.where(r > 0, (dx + 1j * dy) / np.where(r > 0, r, 1.0), 1.0 + 0j)
            out = out * np.tanh(r / xi0) * phase
        return out


def two_vortices(lx: int, ly: int, sep: float, angle_deg: float) -> VortexConfig:
    """Two vortices centred on the lattice, separated by ``sep`` at ``angle_deg``.

    The angle is measured from the +x lattice axis, so 0 deg is the antinodal
    direction and 45 deg the nodal direction of a ``d_{x^2-y^2}`` form factor.
    Centres are deliberately placed on plaquette centres (half-integer
    coordinates) so that no lattice site sits exactly on a singularity.
    """
    ang = np.deg2rad(angle_deg)
    ux, uy = np.cos(ang), np.sin(ang)
    cx, cy = (lx - 1) / 2.0, (ly - 1) / 2.0
    r1 = (cx - 0.5 * sep * ux, cy - 0.5 * sep * uy)
    r2 = (cx + 0.5 * sep * ux, cy + 0.5 * sep * uy)
    return VortexConfig(centers=(r1, r2))


# ------------------------------------------------------------ real-space H_BdG


class Lattice:
    """Site indexing for an ``lx`` by ``ly`` open square lattice."""

    def __init__(self, lx: int, ly: int) -> None:
        self.lx = int(lx)
        self.ly = int(ly)
        self.n = self.lx * self.ly
        self.xs, self.ys = np.meshgrid(
            np.arange(self.lx), np.arange(self.ly), indexing="ij"
        )
        self.xs = self.xs.ravel()
        self.ys = self.ys.ravel()

    def index(self, x, y):
        return np.asarray(x) * self.ly + np.asarray(y)

    def bonds(self, direction: str, periodic: bool = False):
        """Site-index pairs ``(i, j)`` with ``j = i + direction``.

        Returns ``(i, j, xm, ym)`` where ``(xm, ym)`` is the bond midpoint.  With
        ``periodic=True`` the wrap-around bonds are included; their midpoints are
        then meaningless for a vortex profile, so periodic assembly is only used
        for the uniform-Delta consistency test.
        """
        if direction == "x":
            mask = np.ones(self.n, bool) if periodic else self.xs < self.lx - 1
            xi_, yi_ = self.xs[mask], self.ys[mask]
            xj_, yj_ = (xi_ + 1) % self.lx if periodic else xi_ + 1, yi_
        elif direction == "y":
            mask = np.ones(self.n, bool) if periodic else self.ys < self.ly - 1
            xi_, yi_ = self.xs[mask], self.ys[mask]
            xj_, yj_ = xi_, (yi_ + 1) % self.ly if periodic else yi_ + 1
        else:
            raise ValueError(direction)
        i = self.index(xi_, yi_)
        j = self.index(xj_, yj_)
        return i, j, 0.5 * (xi_ + xj_), 0.5 * (yi_ + yj_)


def _block_entries(rows, cols, i, j, block, data_i, data_j, data_v):
    """Accumulate a dense ``block`` into COO lists at Nambu site-pair (i, j)."""
    nb = block.shape[0]
    for a in range(nb):
        for b in range(nb):
            v = block[a, b]
            if v == 0:
                continue
            data_i.append(nb * np.asarray(i) + a)
            data_j.append(nb * np.asarray(j) + b)
            data_v.append(np.full(np.shape(i), v, dtype=complex))


def build_hamiltonian(
    model: Model,
    lx: int,
    ly: int,
    vortices: VortexConfig | None = None,
    mu_map: np.ndarray | None = None,
    periodic: bool = False,
) -> sp.csr_matrix:
    """Assemble the sparse real-space BdG Hamiltonian with open boundaries.

    Parameters
    ----------
    model : the model parameters.
    lx, ly : lattice dimensions.
    vortices : vortex configuration; ``None`` means a uniform pair potential.
    mu_map : optional per-site chemical potential of shape ``(lx*ly,)``, used to
        drive a boundary shell into the trivial phase.  Overrides ``model.mu``.

    Returns
    -------
    A Hermitian ``csr_matrix`` of dimension ``4 * lx * ly``.
    """
    lat = Lattice(lx, ly)
    n = lat.n
    di: list = []
    dj: list = []
    dv: list = []

    vortices = vortices or VortexConfig()

    def profile(x, y):
        return vortices.order_parameter(x, y, model.xi0)

    # ---- on-site ---------------------------------------------------------
    site = np.arange(n)
    mu_site = np.full(n, model.mu, dtype=float) if mu_map is None else np.asarray(
        mu_map, dtype=float
    )
    if mu_site.shape != (n,):
        raise ValueError("mu_map must have shape (lx*ly,)")

    # normal block: (4t - mu) s0 + Vz sz   (xi(k) = 2t(2 - cos kx - cos ky) - mu)
    onsite_const = 4.0 * model.t
    for a in range(2):
        for b in range(2):
            base = onsite_const * S0[a, b] + model.vz * SZ[a, b]
            if base != 0 or a == b:
                vals = base - (mu_site if a == b else 0.0)
                if np.any(vals != 0):
                    di.append(4 * site + a)
                    dj.append(4 * site + b)
                    dv.append(np.asarray(vals, dtype=complex))
            # hole block: -h^*  ->  -(base - mu)^*
            base_h = -np.conj(onsite_const * S0[a, b] + model.vz * SZ[a, b])
            vals_h = base_h + (mu_site if a == b else 0.0)
            if np.any(vals_h != 0):
                di.append(4 * site + 2 + a)
                dj.append(4 * site + 2 + b)
                dv.append(np.asarray(vals_h, dtype=complex))

    # on-site pairing (s-wave and the constant part of extended-s)
    onsite_pair = {"s": model.d0, "ext-s": model.d0 * model.ds, "d": 0.0}[model.form]
    onsite_pair = onsite_pair + 1j * model.d_is
    if onsite_pair != 0.0:
        prof = profile(lat.xs, lat.ys)
        for a in range(2):
            for b in range(2):
                if I_SY[a, b] == 0:
                    continue
                vals = onsite_pair * I_SY[a, b] * prof
                di.append(4 * site + a)
                dj.append(4 * site + 2 + b)
                dv.append(vals)
                di.append(4 * site + 2 + b)
                dj.append(4 * site + a)
                dv.append(np.conj(vals))

    # ---- bonds -----------------------------------------------------------
    # bond pairing coefficients (coefficient of the (+n) and (-n) bond)
    if model.form == "ext-s":
        pair_bond = {"x": model.d0 * 0.25, "y": model.d0 * 0.25}
    elif model.form == "d":
        pair_bond = {"x": model.d0 * 0.25, "y": -model.d0 * 0.25}
    else:
        pair_bond = {"x": 0.0, "y": 0.0}

    for direction in ("x", "y"):
        i, j, xm, ym = lat.bonds(direction, periodic=periodic)

        # normal hopping: -t s0 + Rashba
        #   -alpha sin kx sy  -> (+x) block  = (-alpha sy)/(2i) = (i alpha/2) sy
        #   +alpha sin ky sx  -> (+y) block  = ( alpha sx)/(2i) = (-i alpha/2) sx
        hop = -model.t * S0
        if direction == "x":
            hop = hop + 0.5j * model.alpha * SY
        else:
            hop = hop - 0.5j * model.alpha * SX

        for a in range(2):
            for b in range(2):
                v = hop[a, b]
                if v == 0:
                    continue
                # particle block, i -> j and the Hermitian partner
                di.append(4 * i + a)
                dj.append(4 * j + b)
                dv.append(np.full(i.shape, v, dtype=complex))
                di.append(4 * j + b)
                dj.append(4 * i + a)
                dv.append(np.full(i.shape, np.conj(v), dtype=complex))
                # hole block carries -h^*
                di.append(4 * i + 2 + a)
                dj.append(4 * j + 2 + b)
                dv.append(np.full(i.shape, -np.conj(v), dtype=complex))
                di.append(4 * j + 2 + b)
                dj.append(4 * i + 2 + a)
                dv.append(np.full(i.shape, -v, dtype=complex))

        c = pair_bond[direction]
        if c != 0.0:
            prof = profile(xm, ym)
            for a in range(2):
                for b in range(2):
                    if I_SY[a, b] == 0:
                        continue
                    # symmetric (singlet) bond pairing: same value on i->j and j->i
                    vals = c * I_SY[a, b] * prof
                    di.append(4 * i + a)
                    dj.append(4 * j + 2 + b)
                    dv.append(vals)
                    di.append(4 * j + a)
                    dj.append(4 * i + 2 + b)
                    dv.append(vals)
                    # Hermitian conjugates
                    di.append(4 * j + 2 + b)
                    dj.append(4 * i + a)
                    dv.append(np.conj(vals))
                    di.append(4 * i + 2 + b)
                    dj.append(4 * j + a)
                    dv.append(np.conj(vals))

    rows = np.concatenate([np.asarray(x).ravel() for x in di])
    cols = np.concatenate([np.asarray(x).ravel() for x in dj])
    vals = np.concatenate([np.asarray(x).ravel() for x in dv])
    h = sp.coo_matrix((vals, (rows, cols)), shape=(4 * n, 4 * n)).tocsr()
    h.sum_duplicates()
    return h


def trivial_shell_mu(model: Model, lx: int, ly: int, width: float, mu_out: float
                     ) -> np.ndarray:
    """A ``mu_map`` that ramps to ``mu_out`` within ``width`` of the boundary.

    Driving the boundary shell far outside the band suppresses the amplitude of
    the low-energy edge states there.  It does not remove the chiral edge mode
    (a topological/trivial interface always carries one), but it moves that
    interface inward and keeps the sample edge inert, which is useful when
    checking that results are insensitive to the boundary treatment.
    """
    lat = Lattice(lx, ly)
    d = np.minimum.reduce(
        [lat.xs, lat.ys, (lx - 1) - lat.xs, (ly - 1) - lat.ys]
    ).astype(float)
    w = max(width, 1e-9)
    ramp = np.clip((w - d) / w, 0.0, 1.0)
    return model.mu + (mu_out - model.mu) * ramp
