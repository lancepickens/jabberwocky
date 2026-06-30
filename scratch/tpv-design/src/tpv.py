"""
Detailed-balance thermophotovoltaic (TPV) design study for an 1800 C emitter.

Physics
-------
Emitter: graybody at T_e = 1800 C = 2073.15 K, emissivity eps_e.
Cell:    at T_c = 300 K, operated in the radiative (Shockley-Queisser) limit
         with EQE = 1 above the band gap and a sub-bandgap back-surface
         reflector (BSR) of reflectivity R_sub that recycles below-gap photons
         back to the emitter (the key to modern >40% TPV).

Spectral photon flux into the cell from a blackbody at T with chemical
potential mu (J/photon-area-energy):
    n(E) = (2*pi / (h^3 c^2)) * E^2 / (exp((E-mu)/kT) - 1)
Power spectral density: p(E) = E * n(E).

Net heat the cell must be "paid for" (TPV system efficiency denominator):
    Q_in = [eps * P_emit(>Eg) - P_cell_emit(>Eg, V)]      (above-gap net)
           + (1 - R_sub) * eps * P_emit(<Eg)              (parasitic sub-gap)
Electrical output: P_out = J * V,  J = q*(Phi_emit(>Eg) - Phi_cell(>Eg,V)).
Efficiency: eta = P_out / Q_in.
"""

import numpy as np

# ---- physical constants (SI) ----
h = 6.62607015e-34
c = 2.99792458e8
kB = 1.380649e-23
q = 1.602176634e-19
sigma = 5.670374419e-8

T_E = 1800.0 + 273.15      # emitter temperature, K
T_C = 300.0                # cell temperature, K
EPS_E = 1.0                # emitter emissivity (selective/graybody); 1 = blackbody

# fine energy grid in eV
E_eV = np.linspace(0.02, 6.0, 60000)
E_J = E_eV * q
dE_J = np.gradient(E_J)

PREF = 2.0 * np.pi / (h**3 * c**2)   # hemispherical flux prefactor


def photon_flux_density(E_J, T, mu_J=0.0):
    """Spectral photon flux n(E) [photons / m^2 / s / J]."""
    x = (E_J - mu_J) / (kB * T)
    # guard against overflow; large x -> ~exp(-x)
    out = np.zeros_like(E_J)
    small = x < 700
    out[small] = PREF * E_J[small]**2 / np.expm1(x[small])
    return out


def band_integrals(T, mu_J, Elo_eV, Ehi_eV=np.inf):
    """Return (photon flux, power flux) integrated over [Elo, Ehi] for a
    blackbody at T with chemical potential mu_J."""
    mask = (E_eV >= Elo_eV) & (E_eV <= (Ehi_eV if np.isfinite(Ehi_eV) else E_eV.max()+1))
    n = photon_flux_density(E_J, T, mu_J)
    phi = np.sum(n[mask] * dE_J[mask])              # photons / m^2 / s
    pwr = np.sum(n[mask] * E_J[mask] * dE_J[mask])  # W / m^2
    return phi, pwr


def cell_band_constants(Elo_eV, Ehi_eV=np.inf):
    """Boltzmann-limit cell emission constants for band [Elo,Ehi] at T_C.
    At 300 K with Eg>0.4 eV, (E-qV)>>kT so exp-1 ~ exp; emission scales as
    exp(qV/kT_c). Returns (phi0, p0) s.t. phi_cell(V)=phi0*exp(qV/kT_c),
    p_cell(V)=p0*exp(qV/kT_c). Error vs exact Bose form < 0.01%."""
    hi = Ehi_eV if np.isfinite(Ehi_eV) else E_eV.max() + 1
    mask = (E_eV >= Elo_eV) & (E_eV <= hi)
    boltz = PREF * E_J[mask]**2 * np.exp(-E_J[mask] / (kB * T_C))
    phi0 = np.sum(boltz * dE_J[mask])
    p0 = np.sum(boltz * E_J[mask] * dE_J[mask])
    return phi0, p0


def single_junction(Eg, R_sub=1.0, eps=EPS_E, Ehi=np.inf):
    """Detailed-balance TPV for one cell with gap Eg (eV).
    Returns dict with optimal V, J, P_out, Q_in, efficiency, power density."""
    # emitter-supplied above-gap photon flux & power (graybody)
    phi_emit, p_emit_above = band_integrals(T_E, 0.0, Eg, Ehi)
    phi_emit *= eps
    p_emit_above *= eps
    # sub-bandgap power from emitter (parasitically absorbed fraction)
    _, p_emit_sub = band_integrals(T_E, 0.0, 0.0, Eg)
    p_emit_sub *= eps

    # sweep cell voltage to maximize electrical power
    Vs = np.linspace(0.0, Eg - 1e-3, 800)
    best = None
    for V in Vs:
        mu = q * V
        phi_cell, p_cell_above = band_integrals(T_C, mu, Eg, Ehi)
        J = q * (phi_emit - phi_cell)               # A / m^2
        if J <= 0:
            continue
        P_out = J * V
        Q_in = (p_emit_above - p_cell_above) + (1.0 - R_sub) * p_emit_sub
        eta = P_out / Q_in if Q_in > 0 else 0.0
        if (best is None) or (P_out * eta > best["fom"]):
            # select operating point maximizing power; track eta there
            pass
        if (best is None) or (P_out > best["P_out"]):
            best = dict(V=V, J=J, P_out=P_out, Q_in=Q_in, eta=eta,
                        fom=P_out, Pden=P_out)
    if best is None:
        return dict(Eg=Eg, V=0, J=0, P_out=0, Q_in=p_emit_sub, eta=0, Pden=0)
    best["Eg"] = Eg
    return best


def single_junction_max_eff(Eg, R_sub=1.0, eps=EPS_E, Ehi=np.inf):
    """Same as single_junction but pick V that maximizes efficiency
    (the TPV-relevant objective when sub-gap photons are recycled)."""
    phi_emit, p_emit_above = band_integrals(T_E, 0.0, Eg, Ehi)
    phi_emit *= eps; p_emit_above *= eps
    _, p_emit_sub = band_integrals(T_E, 0.0, 0.0, Eg)
    p_emit_sub *= eps
    phi0, p0 = cell_band_constants(Eg, Ehi)
    Vs = np.linspace(0.0, Eg - 1e-3, 2000)
    g = np.exp(q * Vs / (kB * T_C))
    J = q * (phi_emit - phi0 * g)
    P_out = J * Vs
    Q_in = (p_emit_above - p0 * g) + (1.0 - R_sub) * p_emit_sub
    eta = np.where((J > 0) & (Q_in > 0), P_out / Q_in, 0.0)
    i = int(np.argmax(eta))
    return dict(Eg=Eg, V=float(Vs[i]), J=float(J[i]), P_out=float(P_out[i]),
                Q_in=float(Q_in[i]), eta=float(eta[i]), Pden=float(P_out[i]))


def tandem_series(Eg1, Eg2, R_sub=1.0, eps=EPS_E):
    """Two-junction series-connected stack, top gap Eg1 > bottom gap Eg2.
    Top absorbs >Eg1, bottom absorbs Eg2..Eg1. Series => current matched,
    voltages add. Optimize over matched current. Returns efficiency dict."""
    if Eg1 <= Eg2:
        return None
    # emitter photon supply per subcell window
    phi1, p1_above = band_integrals(T_E, 0.0, Eg1)      # top: >Eg1
    phi2, p2_band = band_integrals(T_E, 0.0, Eg2, Eg1)  # bottom: Eg2..Eg1
    phi1 *= eps; p1_above *= eps; phi2 *= eps; p2_band *= eps
    _, p_emit_sub = band_integrals(T_E, 0.0, 0.0, Eg2)
    p_emit_sub *= eps
    # analytic cell emission constants per subcell band
    phi0_1, p0_1 = cell_band_constants(Eg1, np.inf)
    phi0_2, p0_2 = cell_band_constants(Eg2, Eg1)

    # subcell current as a function of its own voltage (vectorized):
    #   J_i(V) = q*(phi_i - phi0_i*exp(qV/kT_c))
    def Jcurve(Vs, phi_emit, phi0):
        return q * (phi_emit - phi0 * np.exp(q * Vs / (kB * T_C)))

    V1g = np.linspace(0.0, Eg1 - 1e-3, 2000)
    V2g = np.linspace(0.0, Eg2 - 1e-3, 2000)
    J1c = Jcurve(V1g, phi1, phi0_1)
    J2c = Jcurve(V2g, phi2, phi0_2)

    Jmax = min(q * phi1, q * phi2)
    Jgrid = np.linspace(0.02 * Jmax, 0.9995 * Jmax, 400)
    best = None
    for J in Jgrid:
        # invert each subcell J-V to find operating voltage at matched current
        m1 = J1c >= J; m2 = J2c >= J
        if not m1.any() or not m2.any():
            continue
        V1 = V1g[m1][-1]; V2 = V2g[m2][-1]   # highest V still delivering >=J
        g1 = np.exp(q * V1 / (kB * T_C)); g2 = np.exp(q * V2 / (kB * T_C))
        P_out = J * (V1 + V2)
        Q_in = (p1_above - p0_1 * g1) + (p2_band - p0_2 * g2) + (1.0 - R_sub) * p_emit_sub
        eta = P_out / Q_in if Q_in > 0 else 0
        if (best is None) or (eta > best["eta"]):
            best = dict(Eg1=Eg1, Eg2=Eg2, V1=float(V1), V2=float(V2), J=float(J),
                        P_out=float(P_out), Q_in=float(Q_in), eta=float(eta), Pden=float(P_out))
    return best


if __name__ == "__main__":
    # quick sanity: total emitted power vs Stefan-Boltzmann
    _, p_tot = band_integrals(T_E, 0.0, 0.0)
    print(f"Emitter total radiant exitance: {p_tot/1e4:.2f} W/cm^2 "
          f"(sigma T^4 = {sigma*T_E**4/1e4:.2f} W/cm^2)")
    lam_peak = 2897.771955e-6 / T_E
    print(f"Wien peak wavelength: {lam_peak*1e6:.3f} um  "
          f"(photon energy {1.239841984/(lam_peak*1e6):.3f} eV)")
