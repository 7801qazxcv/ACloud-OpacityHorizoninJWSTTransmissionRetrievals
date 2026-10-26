"""
stage16_fig_robustness.py  --  figures for Section 6 (revised robustness)
=========================================================================
fig7_horizon_vs_state.pdf
  (a) cloud-opacity horizon against the ASSUMED metallicity of the fiducial
      atmosphere, for WASP-39 b G395H (stage14_refine.json) and for GJ 486 b
      (stage14_refine.json, rows from stage11 Part C, with the mean molecular
      weight labelled).  Shows explicitly how far the horizon travels when the
      assumed state changes -- the honest bound on "universality".
  (b) horizon against the assumed absorber set (stage13_chemistry.json): the
      same spectrum and forward model with individual species removed.

Every point is a measured diagnostic; nothing here is simulated.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

import figstyle
figstyle.unify_fonts()
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "savefig.dpi": 300, "legend.frameon": False,
                     "axes.linewidth": 0.9})

CW, DW = 3.5, 7.2


def main():
    d14 = json.load(open(os.path.join(HERE, "stage14_refine.json"), encoding="utf-8"))
    d13 = json.load(open(os.path.join(HERE, "stage13_chemistry.json"), encoding="utf-8"))
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(DW, 3.9),
                                   gridspec_kw=dict(width_ratios=[1.1, 1.1]))

    # ---------------- (a) horizon vs assumed metallicity ------------------
    rows = [r for r in d14["fiducial"] if r.get("refined") is not None]
    lz = np.array([r["fiducial"]["logZ"] for r in rows])
    hh = np.array([r["refined"] for r in rows])
    jitter = np.array([r["fiducial"]["T"] for r in rows])
    axa.scatter(lz - 0.06, hh, s=34, marker="o", facecolor=figstyle.FAMILIES["WASP-39"][0], alpha=0.8, edgecolor="k",
                lw=0.4, zorder=3, label="WASP-39 b (14 states)")
    mean_by_z = [(z, hh[lz == z].mean(), hh[lz == z].min(), hh[lz == z].max())
                 for z in sorted(set(lz))]
    axa.errorbar([z + 0.06 for z, *_ in mean_by_z], [m for _, m, _, _ in mean_by_z],
                 yerr=[[m - lo for _, m, lo, _ in mean_by_z],
                       [hi - m for _, m, _, hi in mean_by_z]],
                 fmt="s", ms=5, capsize=3, color=figstyle.FAMILIES["WASP-39"][3], mec="k", mew=0.4,
                 lw=1.0, zorder=4, label="WASP-39 b (binned mean)")
    gj = [r for r in d14["gj486"] if r.get("horizon_refined") is not None]
    axa.plot([r["logZ"] for r in gj], [r["horizon_refined"] for r in gj],
             "-^", ms=6, lw=1.4, color=figstyle.FAMILIES["GJ 486"][0], mec="k", mew=0.4, zorder=5,
             label="GJ 486 b (3 μ)")
    for r in gj:
        axa.annotate(r"$\mu$=%.1f" % r["mu_mean_amu"],
                     (r["logZ"], r["horizon_refined"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8.2,
                     color=figstyle.FAMILIES["GJ 486"][1])
    axa.set_xlabel(r"assumed fiducial metallicity $\log Z_{\rm fid}$ [dex, solar]")
    axa.set_ylabel(r"horizon  $\log_{10}(P_{\rm c}^{\rm crit}/{\rm Pa})$")
    axa.set_title("(a) the horizon depends on the assumed state",
                  fontsize=figstyle.FONTSIZE - 1, pad=8)
    figstyle.tidy_axes(axa)

    # ---------------- (b) horizon vs absorber set -------------------------
    labels = {"baseline": "all species", "no_H2O": "no H$_2$O",
              "no_CO2": "no CO$_2$", "no_CH4_CO": "no CH$_4$, CO",
              "no_H2O_CO2": "no H$_2$O, CO$_2$", "only_H2O": "H$_2$O only"}
    rowsb = [r for r in d13["rows"] if r["horizon_logPc"] is not None]
    xs = np.arange(len(rowsb))
    vals = [r["horizon_logPc"] for r in rowsb]
    axb.bar(xs, vals, color=[figstyle.FAMILIES["WASP-39"][0]
                             if r["variant"] == "baseline"
                             else figstyle.FAMILIES["WASP-39"][3]
                             for r in rowsb], alpha=0.85,
                    edgecolor="k", linewidth=0.5)
    axb.axhline(rowsb[0]["horizon_logPc"], ls=":", lw=0.9, color="k")
    axb.set_xticks(xs)
    # two-line labels keep the double removals unambiguous without crowding
    short_lab = {"all species": "all", "no H$_2$O": "$-$H2O",
                 "no CO$_2$": "$-$CO2", "no CH$_4$, CO": "$-$CH4\n$-$CO",
                 "no H$_2$O, CO$_2$": "$-$H2O\n$-$CO2", "H$_2$O only": "only\nH2O"}
    axb.set_xticklabels([short_lab.get(labels.get(r["variant"], r["variant"]),
                                       labels.get(r["variant"], r["variant"]))
                         for r in rowsb], rotation=0, ha="center", fontsize=8.0)
    axb.set_ylabel(r"horizon  $\log_{10}(P_{\rm c}^{\rm crit}/{\rm Pa})$")
    axb.set_title("(b) removing H$_2$O opacity moves it by 0.36 dex",
                  fontsize=figstyle.FONTSIZE - 1, pad=8)
    axb.set_ylim(3.2, 4.1)
    fig.tight_layout(pad=0.6)
    figstyle.tidy_axes(axb, minor=False)
    # Legend goes INSIDE panel (a): every off-panel position either hit the
    # title (above) or cost most of a row of text (below).  The upper-left
    # corner of panel (a) is empty for these data.
    handles, labels_ = axa.get_legend_handles_labels()
    axa.legend(handles, labels_, loc="upper right", frameon=True,
               framealpha=0.92, edgecolor="0.6", fontsize=9.0,
               handlelength=1.2, columnspacing=0.8, labelspacing=0.25,
               borderpad=0.35)
    fig.subplots_adjust(top=0.94, bottom=0.10)
    figstyle.save(fig, os.path.join(FIG, "fig7_horizon_vs_state.pdf"))
    print("wrote", os.path.join(FIG, "fig7_horizon_vs_state.pdf"))
    print("WASP-39b horizon range across fiducials: %.3f - %.3f (n=%d)"
          % (min(hh), max(hh), len(hh)))
    print("absorber-set horizons:", [(labels.get(r["variant"], r["variant"]),
                                      round(r["horizon_logPc"], 3)) for r in rowsb])


if __name__ == "__main__":
    main()
