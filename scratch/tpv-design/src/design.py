"""Run the TPV design study and emit results + figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tpv import (band_integrals, photon_flux_density, single_junction_max_eff,
                 tandem_series, E_eV, E_J, T_E, T_C, sigma, q)

OUT = "../out"

# ---- known TPV-relevant semiconductors (band gap in eV at ~300 K) ----
MATERIALS = {
    "Si":                1.12,
    "GaAs":              1.42,
    "InP":               1.34,
    "Ge":                0.66,
    "GaSb":              0.726,
    "In0.53Ga0.47As":    0.74,
    "InGaAs (0.6 eV)":   0.60,
    "InGaAsSb":          0.55,
    "InGaAs (0.50 eV)":  0.50,
    "InAsSbP/InAs":      0.40,
}

R_SUB = 0.97   # realistic gold back-surface reflector for sub-gap photons
R_IDEAL = 1.0  # perfect photon recycling (design ceiling)

print("="*78)
print(f"TPV DESIGN STUDY  |  Emitter {T_E-273.15:.0f} C ({T_E:.1f} K), Cell {T_C:.0f} K")
_, p_tot = band_integrals(T_E, 0.0, 0.0)
print(f"Emitter exitance {p_tot/1e4:.1f} W/cm^2 | Wien peak 1.40 um / 0.887 eV")
print("="*78)

# ---- 1. Single-junction screening of known materials ----
print("\n[1] KNOWN-MATERIAL SINGLE-JUNCTION SCREEN (perfect & R_sub=0.97 BSR)")
print(f"{'material':<18}{'Eg(eV)':>7}{'eta_ideal':>11}{'eta_0.97':>10}"
      f"{'Pden(W/cm2)':>13}{'Voc(V)':>8}")
rows = []
for name, Eg in sorted(MATERIALS.items(), key=lambda kv: kv[1]):
    bi = single_junction_max_eff(Eg, R_sub=R_IDEAL)
    br = single_junction_max_eff(Eg, R_sub=R_SUB)
    rows.append((name, Eg, bi["eta"], br["eta"], br["Pden"]/1e4, br["V"]))
    print(f"{name:<18}{Eg:>7.3f}{bi['eta']*100:>10.1f}%{br['eta']*100:>9.1f}%"
          f"{br['Pden']/1e4:>13.2f}{br['V']:>8.3f}")

# ---- 2. Continuous single-junction gap sweep -> optimum ----
print("\n[2] CONTINUOUS BAND-GAP SWEEP")
Egs = np.linspace(0.30, 1.40, 56)
eta_ideal = np.array([single_junction_max_eff(Eg, R_sub=R_IDEAL)["eta"] for Eg in Egs])
eta_real  = np.array([single_junction_max_eff(Eg, R_sub=R_SUB)["eta"]  for Eg in Egs])
pden      = np.array([single_junction_max_eff(Eg, R_sub=R_SUB)["Pden"]/1e4 for Eg in Egs])
i_ideal = int(np.argmax(eta_ideal)); i_real = int(np.argmax(eta_real))
print(f"  Optimal gap (perfect BSR): {Egs[i_ideal]:.3f} eV -> eta {eta_ideal[i_ideal]*100:.1f}%")
print(f"  Optimal gap (R_sub=0.97):  {Egs[i_real]:.3f} eV -> eta {eta_real[i_real]*100:.1f}%")

# ---- 3. Two-junction tandem optimization ----
print("\n[3] TWO-JUNCTION SERIES TANDEM OPTIMIZATION (perfect BSR)")
best_tan = None
grid = np.linspace(0.45, 1.30, 26)
for Eg1 in grid:
    for Eg2 in grid:
        if Eg2 >= Eg1 - 0.12:
            continue
        r = tandem_series(Eg1, Eg2, R_sub=R_IDEAL)
        if r and (best_tan is None or r["eta"] > best_tan["eta"]):
            best_tan = r
print(f"  Best tandem: top {best_tan['Eg1']:.2f} eV / bottom {best_tan['Eg2']:.2f} eV")
print(f"     eta {best_tan['eta']*100:.1f}%  |  Pden {best_tan['Pden']/1e4:.2f} W/cm^2"
      f"  |  Voc_sum {best_tan['V1']+best_tan['V2']:.2f} V")
br97 = tandem_series(best_tan['Eg1'], best_tan['Eg2'], R_sub=R_SUB)
print(f"     with R_sub=0.97: eta {br97['eta']*100:.1f}%")

# tandem optimized FOR the realistic reflector (sub-gap leakage punishes high gaps)
best_tan_r = None
for Eg1 in grid:
    for Eg2 in grid:
        if Eg2 >= Eg1 - 0.12:
            continue
        r = tandem_series(Eg1, Eg2, R_sub=R_SUB)
        if r and (best_tan_r is None or r["eta"] > best_tan_r["eta"]):
            best_tan_r = r
print(f"  Best tandem @R_sub=0.97: top {best_tan_r['Eg1']:.2f} / bottom "
      f"{best_tan_r['Eg2']:.2f} eV -> eta {best_tan_r['eta']*100:.1f}%  "
      f"Pden {best_tan_r['Pden']/1e4:.1f} W/cm^2")

# single-junction optimum power density at the realistic optimum
sj_opt = single_junction_max_eff(0.86, R_sub=R_SUB)
print(f"  Single-junction @0.86 eV, R=0.97: eta {sj_opt['eta']*100:.1f}%  "
      f"Pden {sj_opt['Pden']/1e4:.1f} W/cm^2")

# realistic-material tandem: GaAs/GaSb-class and InGaAs/InGaAs (Nature 2022 style)
print("\n  Buildable tandem candidates (perfect BSR / R_sub=0.97):")
for nm, (g1, g2) in {
    "GaAs(1.42)/GaSb(0.73)":      (1.42, 0.726),
    "1.2/0.95 (AlGaAs/GaAs-cl)":  (1.20, 0.95),
    "1.4/1.2 (Nature'22 hi-T)":   (1.40, 1.20),
    "In0.53GaAs(0.74)/0.55":      (0.74, 0.55),
}.items():
    a = tandem_series(g1, g2, R_sub=R_IDEAL)
    b = tandem_series(g1, g2, R_sub=R_SUB)
    print(f"   {nm:<28} eta {a['eta']*100:5.1f}% / {b['eta']*100:5.1f}%"
          f"   Pden {b['Pden']/1e4:5.2f} W/cm^2")

# ---- 4. Back-surface-reflector sensitivity for the recommended design ----
print("\n[4] SUB-BANDGAP REFLECTOR SENSITIVITY (recommended 0.74/0.55 eV tandem)")
print(f"  {'R_sub':>7}{'eta':>9}{'Pden(W/cm2)':>14}")
Rs = [0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 1.0]
refl_eta = []
for R in Rs:
    r = tandem_series(0.74, 0.55, R_sub=R)
    refl_eta.append(r["eta"]*100)
    print(f"  {R:>7.3f}{r['eta']*100:>8.1f}%{r['Pden']/1e4:>14.1f}")
# same sweep for a high-gap tandem to show its reflector sensitivity
refl_eta_hi = [tandem_series(1.30, 1.13, R_sub=R)["eta"]*100 for R in Rs]

# ============================ FIGURES ============================
# Fig A: blackbody spectrum + candidate gaps
fig, ax = plt.subplots(figsize=(9, 5.2))
n_e = photon_flux_density(E_J, T_E, 0.0)
p_e = n_e * E_J / q            # spectral power per eV (arb scaled)
ax.fill_between(E_eV, p_e/p_e.max(), color="#d9772b", alpha=0.35, label=f"{T_E-273.15:.0f} C blackbody")
ax.plot(E_eV, p_e/p_e.max(), color="#b85c12", lw=1.5)
for nm, Eg in MATERIALS.items():
    ax.axvline(Eg, ls="--", lw=0.8, color="#333", alpha=0.5)
ax.axvline(Egs[i_ideal], color="crimson", lw=2.2, label=f"optimal single gap {Egs[i_ideal]:.2f} eV")
ax.axvline(0.887, color="navy", lw=1.2, ls=":", label="Wien peak 0.887 eV")
ax.set_xlim(0, 2.2); ax.set_xlabel("photon energy (eV)")
ax.set_ylabel("normalized spectral power")
ax.set_title("1800 C blackbody spectrum and candidate band gaps")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/fig_spectrum.png", dpi=130)

# Fig B: efficiency vs band gap
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(Egs, eta_ideal*100, "-o", ms=3, color="crimson", label="perfect BSR (recycling)")
ax.plot(Egs, eta_real*100, "-s", ms=3, color="darkorange", label="R_sub = 0.97 (Au reflector)")
ax2 = ax.twinx()
ax2.plot(Egs, pden, "--", color="steelblue", label="power density")
ax2.set_ylabel("power density (W/cm^2)", color="steelblue")
for nm, Eg in MATERIALS.items():
    if 0.3 <= Eg <= 1.4:
        ax.annotate(nm.split()[0], (Eg, 4), rotation=90, fontsize=6,
                    ha="center", va="bottom", color="#444")
ax.axvline(Egs[i_ideal], color="crimson", ls=":", lw=1)
ax.set_xlabel("band gap (eV)"); ax.set_ylabel("TPV efficiency (%)")
ax.set_title("Single-junction TPV efficiency vs band gap (1800 C emitter)")
ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
fig.savefig(f"{OUT}/fig_efficiency.png", dpi=130)

# Fig C: reflector sensitivity
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([r*100 for r in Rs], refl_eta, "-o", color="seagreen",
        label="low-gap tandem 0.74/0.55 eV (high P-density)")
ax.plot([r*100 for r in Rs], refl_eta_hi, "-s", color="indianred",
        label="high-gap tandem 1.30/1.13 eV")
ax.set_xlabel("sub-bandgap reflectivity R_sub (%)")
ax.set_ylabel("TPV efficiency (%)")
ax.set_title("Why photon recycling dominates TPV efficiency at 1800 C")
ax.grid(alpha=0.3); ax.legend(fontsize=9); fig.tight_layout()
fig.savefig(f"{OUT}/fig_reflector.png", dpi=130)

print(f"\nFigures written to {OUT}/fig_spectrum.png, {OUT}/fig_efficiency.png, {OUT}/fig_reflector.png")

# save machine-readable summary
import json
summary = dict(
    emitter_C=T_E-273.15, emitter_K=T_E, cell_K=T_C,
    exitance_Wcm2=p_tot/1e4, wien_eV=0.887,
    single_opt_ideal=dict(Eg=float(Egs[i_ideal]), eta=float(eta_ideal[i_ideal])),
    single_opt_real=dict(Eg=float(Egs[i_real]), eta=float(eta_real[i_real])),
    materials=[dict(name=r[0], Eg=r[1], eta_ideal=r[2], eta_097=r[3], Pden_Wcm2=r[4], V=r[5]) for r in rows],
    tandem_best=dict(Eg1=best_tan['Eg1'], Eg2=best_tan['Eg2'], eta_ideal=best_tan['eta'],
                     eta_097=br97['eta'], Pden_Wcm2=best_tan['Pden']/1e4),
)
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Wrote", f"{OUT}/summary.json")
