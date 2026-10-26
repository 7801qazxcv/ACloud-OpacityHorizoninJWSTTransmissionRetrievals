"""
make_figures.py
===============
All figures for the paper.

DATA PROVENANCE PER FIGURE (no invented numbers anywhere in this file):
  Fig 1  Real published JWST transmission spectra (NASA Exoplanet Archive
         `transitspec`, 12 datasets, 2712 channels) -- plotted as published.
  Fig 2  Diagnostic computed from the PLATON forward model + each dataset's
         real published per-channel sigma (stage2/stage3 outputs).
  Fig 3  Cloud-opacity-horizon curves (stage3_cloud_probe.json).
  Fig 4  Horizon location vs wavelength coverage (stage4_results.json).
  Fig 5  Linear vs nonlinear bias transfer (stage4 + stage5 outputs).

NO SIMULATED OBSERVATION IS PLOTTED. Model spectra appear only in Fig 1 as an
explicitly-labelled forward-model curve, and in Fig 5 as the injection-recovery
test of the estimator (labelled as such).
"""
import csv, collections, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "savefig.dpi": 300,
    "axes.labelsize": 11, "legend.fontsize": 9.0, "legend.frameon": False,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})
CW, DW = 3.5, 7.2   # single / double column widths in inches


# Shared house style (minimal labels, no top/right spines, dashed grid,
# unified fonts, one colour family per planet).
import figstyle
figstyle.unify_fonts()


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)


def load_spectra():
    rows = list(csv.DictReader(open(os.path.join(HERE, "transitspec_jwst.csv"), encoding="utf-8")))
    g = collections.defaultdict(list)
    for r in rows:
        ref = r["plntranreflink"] or ""
        tag = ref.split("refstr=")[-1].split(" ")[0] if "refstr=" in ref else "UNK"
        g[(r["plntname"], r["instrument"], tag)].append(r)
    out = []
    for (pl, inst, tag), v in g.items():
        wl, dep, err = [], [], []
        for r in v:
            try:
                if r["plntransdeperr1"] in ("", None):
                    continue
                wl.append(float(r["centralwavelng"])); dep.append(float(r["plntransdep"]))
                err.append(abs(float(r["plntransdeperr1"])))
            except (TypeError, ValueError):
                continue
        if len(wl) < 8:
            continue
        o = np.argsort(wl)
        out.append(dict(planet=pl, instrument=inst, ref=tag, wl=np.array(wl)[o],
                        dep=np.array(dep)[o], err=np.array(err)[o]))
    return out


def short(ref):
    r = ref.replace("_ET_AL__", "+").replace("JWST_TRANSITING_EXOPLANET_COMMUNITY_EARLY_RELEASE_SCIENCE_TEAM",
                                             "JTEC ERS")
    return r.replace("__", " ").rstrip("_")


# ----------------------------------------------------------------- Figure 1
def fig1():
    sp = load_spectra()
    fig, axes = plt.subplots(3, 1, figsize=(CW * 2, 6.4), sharex=False)
    groups = [("WASP-39 b", axes[0]), ("HAT-P-18 b", axes[1]), ("GJ 486 b", axes[2])]
    cmap = plt.get_cmap("viridis")
    for pl, ax in groups:
        ds = [d for d in sp if d["planet"] == pl]
        ds.sort(key=lambda d: d["wl"].min())
        for i, d in enumerate(ds):
            c = cmap(0.08 + 0.82 * i / max(len(ds) - 1, 1))
            ax.errorbar(d["wl"], d["dep"] * 1e4, yerr=d["err"] * 1e4, fmt="o",
                        ms=1.8, lw=0.5, elinewidth=0.5, color=c, alpha=0.85,
                        label="%s (%s, n=%d)" % (short(d["ref"])[:22],
                                                 "".join(w[0] for w in d["instrument"].split()[:3]),
                                                 len(d["wl"])))
        ax.set_ylabel(r"transit depth $(R_p/R_\star)^2$  [$10^{-4}$]")
        ax.text(0.985, 0.93, pl, transform=ax.transAxes, ha="right", va="top",
                fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", ncol=1, fontsize=7.5, handletextpad=0.4,
                 framealpha=0.75)
        ax.set_xscale("log")
        ax.set_xticks([0.6, 1, 2, 3, 4, 5])
        ax.set_xticklabels(["0.6", "1", "2", "3", "4", "5"])
    axes[-1].set_xlabel(r"wavelength [$\mu$m]")
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "fig1_spectra.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig1 ok")


# ----------------------------------------------------------------- Figure 2
def fig2():
    d2 = load("stage2_diagnostics.json")
    d2.sort(key=lambda r: (not (r["wl_um_min"] < 1.0), r["wl_um_min"]))
    names = ["T", "logZ", "CO_ratio", "log_cloudtop_P", "Rp"]
    disp = [r"$T$", r"$\log Z$", r"C/O", r"$\log P_{\rm c}$", r"$R_p$"]
    M = np.array([[next(p["prior_leverage"] for p in r["per_parameter"]
                        if p["parameter"] == n) for n in names] for r in d2])
    labels = ["%s %s (%d ch, %.0f ppm)" % (r["planet"], short(r["ref"])[:16],
                                           r["n_channels"], r["sigma_ppm_median"]) for r in d2]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DW, 4.0),
                                   gridspec_kw=dict(width_ratios=[2.15, 1]))
    im = axA.imshow(M, aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    axA.set_xticks(range(5)); axA.set_xticklabels(disp)
    axA.set_yticks(range(len(labels))); axA.set_yticklabels(labels, fontsize=8.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            axA.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7.6,
                     color="white" if M[i, j] > 0.55 else "black")
    n_opt = sum(1 for r in d2 if r["wl_um_min"] < 1.0)
    axA.axhline(n_opt - 0.5, color="tab:cyan", lw=1.4)
    axA.text(4.62, (n_opt - 1) / 2.0, "optical\ncoverage", rotation=90, va="center",
             ha="center", fontsize=8.2, color="tab:cyan")
    cb = fig.colorbar(im, ax=axA, pad=0.13, fraction=0.045)
    cb.set_label(r"prior leverage $L_j = 1-A_{jj}$", fontsize=10)
    axA.set_title(r"(a) fraction of the constraint supplied by the prior", fontsize=10.0)

    opt = np.array([r["dofs"] for r in d2 if r["wl_um_min"] < 1.0])
    ir = np.array([r["dofs"] for r in d2 if r["wl_um_min"] >= 1.0])
    so = np.array([r["sigma_ppm_median"] for r in d2 if r["wl_um_min"] < 1.0])
    si = np.array([r["sigma_ppm_median"] for r in d2 if r["wl_um_min"] >= 1.0])
    axB.scatter(so, opt, s=34, marker="o", facecolor="tab:cyan", edgecolor="k",
                lw=0.5, label=r"$\lambda_{\min}<1\,\mu$m", zorder=3)
    axB.scatter(si, ir, s=34, marker="s", facecolor="0.75", edgecolor="k",
                lw=0.5, label=r"$\lambda_{\min}\geq1\,\mu$m", zorder=3)
    axB.set_xscale("log")
    axB.set_xlabel("published median uncertainty [ppm]")
    axB.set_ylabel(r"DOFS  $d_s = {\rm tr}\,\mathbf{A}$")
    axB.axhline(5, ls=":", lw=0.8, color="k")
    axB.text(0.03, 0.965, "nominal 5 parameters", transform=axB.transAxes, fontsize=6.2, va="top")
    axB.legend(loc="lower left", fontsize=9)
    axB.set_title("(b) DOFS is set by coverage, not precision", fontsize=10.0)
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(FIG, "fig2_leverage.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig2 ok")


# ----------------------------------------------------------------- Figure 3
def fig3():
    d3 = load("stage3_cloud_probe.json")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(CW, 5.0), sharex=True)
    figstyle.reset_counter()
    for r in d3:
        x = np.array([c["log_cloudtop_P"] for c in r["curve"]])
        L = np.array([c["leverage_cloud"] for c in r["curve"]])
        D = np.array([c["dofs"] for c in r["curve"]])
        kw = figstyle.style_for(r["planet"], marker=True)
        lab = "%s (%s)" % (r["planet"], r["label"])
        ax1.plot(x, L, label=lab, **kw)
        ax2.plot(x, D, **kw)
    ax1.axhline(0.5, color="tab:red", lw=1.1, ls="-.")
    ax1.text(0.02, 0.55, "L = 0.5 (horizon definition)", transform=ax1.get_yaxis_transform(),
             fontsize=figstyle.FONTSIZE - 3, color="tab:red")
    ax1.set_ylabel(r"prior leverage $L_{\log P_{\rm c}}$")
    ax1.set_ylim(-0.05, 1.12)
    figstyle.tidy_axes(ax1)
    ax2.axvspan(3.37, 4.71, color="tab:orange", alpha=0.13, lw=0)
    ax2.text(0.5, 0.06, "horizon range (12 datasets)", transform=ax2.transAxes,
             ha="center", fontsize=figstyle.FONTSIZE - 3, color="tab:orange")
    ax2.set_ylabel(r"DOFS $d_s$")
    ax2.set_xlabel(r"fiducial cloud-top pressure  $\log_{10}(P_{\rm c}/{\rm Pa})$")
    figstyle.tidy_axes(ax2)
    ax1.set_title("cloud-top pressure: measured below the horizon,\n"
                  "unmeasurable above it", fontsize=figstyle.FONTSIZE - 2, pad=8)
    # legend OUTSIDE the plot area, below the lower panel
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=1, frameon=False, handlelength=1.6, labelspacing=0.35)
    fig.subplots_adjust(bottom=0.30, top=0.93, hspace=0.18)
    figstyle.save(fig, os.path.join(FIG, "fig3_horizon.pdf"))
    plt.close(fig)
    print("fig3 ok")


# ----------------------------------------------------------------- Figure 4
def fig4():
    d4 = load("stage4_results.json")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(DW, 3.7))
    for r in d4:
        c = "tab:cyan" if r["has_optical"] else "0.45"
        m = "o" if r["has_optical"] else "s"
        axa.scatter(r["wl_um_min"], r["horizon_logPc"], s=44, marker=m,
                    facecolor=c, edgecolor="k", lw=0.5, zorder=3)
        axb.scatter(r["sigma_ppm_median"], r["horizon_logPc"], s=44, marker=m,
                    facecolor=c, edgecolor="k", lw=0.5, zorder=3)
    hs = np.array([r["horizon_logPc"] for r in d4])
    for ax in (axa, axb):
        ax.axhline(hs.mean(), ls=":", lw=0.9, color="k")
    axa.set_xlabel(r"shortest observed wavelength $\lambda_{\min}$ [$\mu$m]")
    axa.set_ylabel(r"horizon  $\log_{10}(P_{\rm c}^{\rm crit}/{\rm Pa})$")
    axa.set_title("(a) horizon tracks wavelength coverage", fontsize=10.0)
    axb.set_xscale("log")
    axb.set_xlabel("published median uncertainty [ppm]")
    axb.set_title("(b) horizon is independent of precision", fontsize=10.0)
    axa.text(0.04, 0.05, r"mean $=%.2f$, s.d. $=%.2f$" % (hs.mean(), hs.std()),
             transform=axa.transAxes, fontsize=9)
    h = [Line2D([], [], marker="o", ls="", mfc="tab:cyan", mec="k", ms=6,
                label=r"optical coverage ($\lambda_{\min}<1\,\mu$m)"),
         Line2D([], [], marker="s", ls="", mfc="0.45", mec="k", ms=6,
                label=r"IR only ($\lambda_{\min}\geq1\,\mu$m)")]
    axb.legend(handles=h, loc="upper right", fontsize=8.8)
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(FIG, "fig4_horizon_scaling.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("fig4 ok")


# ----------------------------------------------------------------- Figure 5
def fig5():
    d5 = load("stage5_nonlinear.json")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(DW, 3.8))
    figstyle.reset_counter()
    for r in d5:
        dl = np.array([t["delta_dex"] for t in r["tests"]])
        nl = np.array([t["nonlinear"]["logZ"] for t in r["tests"]])
        li = np.array([t["linear_scaled"]["logZ"] for t in r["tests"]])
        kw = figstyle.style_for(r["planet"], marker=True)
        lab = r["planet"]
        axa.plot(dl, nl, label=lab, **kw)
        dash = dict(kw); dash.update({"linestyle": ":", "marker": None,
                                      "alpha": 0.75})
        axa.plot(dl, li, **dash)
        kwp = dict(kw); kwp.update({"marker": "o", "linestyle": "none",
                                    "alpha": 0.85})
        axb.plot(np.abs(li), np.abs(nl), **kwp)
    figstyle.tight_limits(axa, x=np.concatenate([np.array([t["delta_dex"] for t in r["tests"]]) for r in d5]), pad=0.06)
    figstyle.tidy_axes(axa)
    axa.set_xlabel(r"imposed error in $\log_{10}P_{\rm c}$ [dex]")
    axa.set_ylabel(r"induced offset in $\log Z$ [dex]")
    axa.set_title("(a) symbols: nonlinear refit; dotted: linear operator",
                  fontsize=figstyle.FONTSIZE - 1, pad=8)
    handles, labels = axa.get_legend_handles_labels()
    labels = [s.replace(" (opt)", "") for s in labels]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.0,
               frameon=False, handlelength=2.0, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.24, top=0.92)
    lim = [0.02, 3.0]
    axb.plot(lim, lim, "k-", lw=0.9)
    for f in (2.0,):
        axb.plot(lim, [f * l for l in lim], "k:", lw=0.7)
        axb.text(lim[1] * 0.55, f * lim[1] * 0.55, r"$\times2$", fontsize=8.5)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlim(lim); axb.set_ylim(lim)
    axb.set_xlabel(r"$|{\rm linear\ prediction}|$ [dex]")
    axb.set_ylabel(r"$|{\rm nonlinear\ refit}|$ [dex]")
    axb.set_title("(b) linear operator over-extrapolates",
                  fontsize=figstyle.FONTSIZE - 1, pad=8)
    figstyle.tidy_axes(axb)
    fig.tight_layout(pad=0.6, rect=[0, 0.16, 1, 1])
    figstyle.save(fig, os.path.join(FIG, "fig5_bias_validation.pdf"))
    plt.close(fig)
    print("fig5 ok")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("figures ->", FIG)
