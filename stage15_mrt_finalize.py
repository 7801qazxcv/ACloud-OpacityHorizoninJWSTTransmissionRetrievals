"""
stage15_mrt_finalize.py  --  finish the machine-readable package
================================================================
* replaces the coarse (0.25-dex-grid) horizon in data/summary.csv with the
  bisected value that the paper quotes (stage4_results.json), matched on
  planet + channel count + median uncertainty;
* writes data/README.txt with a column-by-column description (units and
  provenance), as requested by the AAS data editor;
* prints every number needed for the manuscript's in-paper example table so
  that no value in the paper is transcribed by hand.
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")


def load_stage4():
    rows = json.load(open(os.path.join(HERE, "stage4_results.json"), encoding="utf-8"))
    out = {}
    for r in rows:
        key = (r["planet"].strip(), int(r["n_channels"]),
               round(float(r["sigma_ppm_median"]), 1))
        out[key] = r["horizon_logPc"]
    return out


def main():
    h4 = load_stage4()
    sp = os.path.join(DATA, "summary.csv")
    rows = list(csv.DictReader(open(sp, encoding="utf-8")))
    n_match = 0
    for r in rows:
        key = (r["planet"].strip(), int(r["n_channels"]),
               round(float(r["sigma_median_ppm"]), 1))
        if key in h4:
            r["horizon_logPc"] = "%.4f" % h4[key]
            n_match += 1
        # NOTE: for the three optical sets the Paper quotes the bisected value
        # from stage4; the 0.25-dex grid value from stage12 is < 0.1 dex away.
    print("horizon values taken from stage4 for %d/%d rows" % (n_match, len(rows)))
    with open(sp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---------------- README ----------------
    readme = """Machine-readable tables for:
"A Cloud-Opacity Horizon Sets What JWST Transmission Spectra Can and Cannot Measure"

Author: Tan Shuxin, Fu Sen
Contact: corresponding author Fu Sen (fusen1991@nuaa.edu.cn)

FILES
-----
summary.csv     One row per analysed transmission spectrum (12 rows). Contains
                the identifiers, wavelength coverage, precision, the fiducial
                state used to evaluate the Jacobian, the prior leverage of each
                retrieved parameter at that state, the degrees of freedom for
                signal, and the cloud-opacity horizon located for that spectrum.

horizon_curves.csv  One row per (spectrum, cloud-top pressure) -- 252 rows.
                Contains the prior leverage of the cloud-top pressure,
                L_logPc = 1 - A_jj, and the degrees of freedom for signal,
                d_s = tr A, as a function of the assumed cloud-top pressure.
                These are the twelve curves of Figure 3; only three are drawn
                in the figure.

Both files are comma-separated ASCII with a single header row. Missing values
are empty (the forward model does not converge for those states).

COLUMNS - summary.csv
---------------------
dataset_id        Zero-padded index + planet + instrument abbreviation + channel
                  count, e.g. "07_WASP-39b_Near_n344". Joins to
                  horizon_curves.csv on the same key.
planet            Planet name as recorded in the NASA Exoplanet Archive.
instrument        Instrument / grating or reduction pipeline as recorded in the
                  Archive transitspec table.
reference         First author + year tag of the published analysis.
n_channels        Number of wavelength channels in the spectrum (dimensionless).
wl_min_um         Shortest channel centre, microns.
wl_max_um         Longest channel centre, microns.
has_optical       1 if any channel lies below 1 micron, else 0.
sigma_median_ppm  Median of the published per-channel 1-sigma uncertainties on
                  the transit depth, ppm. NOTE: the diagnostics use every
                  per-channel value, not this median (see Section 3.3 of the
                  paper and Figure 6).
fiducial_T_K      Assumed isothermal temperature at which the Jacobian is
                  evaluated, K. Taken from the Archive equilibrium temperature.
fiducial_logZ_dex Assumed log10 metallicity relative to solar, dex.
fiducial_CO_ratio Assumed C/O ratio.
fiducial_logPc_dex log10 cloud-top pressure for the per-parameter diagnostics,
                  Pa.
L_T, L_logZ, L_CO_ratio, L_logPc, L_Rp
                  Prior leverage (dimensionless, 0-1) of each parameter at the
                  fiducial state above: the fraction of that parameter's
                  constraint supplied by the prior. L = 1 - A_jj with A the
                  averaging kernel (Equation 5 of the paper). L = 0 means the
                  data alone determine the parameter; L = 1 means the posterior
                  restates the prior.
dofs_at_fiducial  Degrees of freedom for signal, d_s = tr A, at the fiducial
                  state (dimensionless, bound above by 5).
horizon_logPc     log10 cloud-top pressure (Pa) at which L_logPc = 0.5: the
                  cloud-opacity horizon for that spectrum. Located by bisection
                  to ~1e-3 dex (these are the values quoted in Section 4.2 and
                  Figure 4).

COLUMNS - horizon_curves.csv
----------------------------
dataset_id        Same key as in summary.csv.
log_cloudtop_P    log10 of the assumed cloud-top pressure, in Pa. Grid:
                  1.50 to 6.50 dex in steps of 0.25 dex.
leverage_log_cloudtop_P
                  Prior leverage of the cloud-top pressure at that state
                  (dimensionless, 0-1).
dofs              Degrees of freedom for signal at that state (dimensionless).

PROVENANCE
----------
Every quantity is computed from published JWST transmission spectra taken from
the NASA Exoplanet Archive table "transitspec" (no data are simulated) using
the published per-channel transit depths d_i as y and the published per-channel
1-sigma errors as the diagonal of S_epsilon. The forward model is PLATON v6.3
(doi:10.3847/1538-4357/aba1e6) with equilibrium chemistry, an isothermal
temperature profile, and a grey cloud deck; stellar and planetary parameters
are those of Table 3 of the paper. The diagnostics use prior standard
deviations of T 150 K, log Z 1.0 dex, C/O 0.30, log10 P_c 1.5 dex and R_p 1%
(Table 2 of the paper). Reproducing scripts:
https://github.com/[handle]/jwst-identifiability
"""
    open(os.path.join(DATA, "README.txt"), "w", encoding="utf-8").write(readme)
    print("wrote README.txt")

    # ---------------- numbers for the in-paper table ----------------
    print()
    print("--- values for the manuscript example table ---")
    print("%-24s %-9s %6s %8s %8s %8s" % ("dataset_id", "opt", "n", "dofs", "L_logPc", "horizon"))
    for r in rows:
        print("%-24s %-9s %6s %8s %8s %8s"
              % (r["dataset_id"], "optical" if int(r["has_optical"]) else "IR-only",
                 r["n_channels"], r["dofs_at_fiducial"], r["L_logPc"], r["horizon_logPc"]))


if __name__ == "__main__":
    main()
