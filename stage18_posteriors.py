"""
stage18_posteriors.py  --  posteriors, corner plots and correlations
=====================================================================
Referee items #18 (show the retrieval results) and #20 (do the OTHER
parameters move, and are there correlations?).

Reads the emcee chains written by stage10_retrieval_long.py (v3) and produces
  ../figures/fig8_corner_coverage.pdf   posterior of the retrieved parameters
        for the IR-only spectrum and the optical+NIR spectrum, overlaid
  ../figures/fig9_corner_cloudfixed.pdf same for the two runs in which the
        cloud-top pressure is frozen 1.5 dex apart
  stage18_posteriors.json              medians, 16-84 intervals, prior
        fractions, distance to the nearest prior edge, and correlations
and prints everything needed to transcribe into the manuscript.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.path.join(HERE, "ck_stage10")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(HERE, "stage18_posteriors.json")

LABEL = {"Rp": r"$R_p$ [$R_{\rm J}$]", "T": r"$T$ [K]", "logZ": r"$\log Z$ [dex]",
         "CO_ratio": r"C/O", "log_cloudtop_P": r"$\log_{10}P_{\rm c}$ [Pa]"}
R_JUP = 7.1492e7

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 13, "axes.linewidth": 0.8,
                     "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "xtick.major.size": 4, "ytick.major.size": 4,
                     "savefig.dpi": 300})


def _enforce_fonts(fig, tick=13, axis=16):
    """corner sets its own (small) tick fonts; force a size that survives the
    downscale when the figure is inserted at the journal's text width."""
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=tick)
        if ax.xaxis.label:
            ax.xaxis.label.set_size(axis)
        if ax.yaxis.label:
            ax.yaxis.label.set_size(axis)
    return fig


def load(label):
    import stage10_retrieval_long as S
    cf = os.path.join(CK, label + "_chain.npy")
    if not os.path.exists(cf):
        return None
    ch = np.load(cf)
    meta = json.load(open(os.path.join(HERE, "stage10_%s.json" % label),
                          encoding="utf-8"))
    nb = meta["burn_in_steps"]
    post = ch[nb:].reshape(-1, ch.shape[2])
    return dict(post=post, meta=meta)


def main():
    import corner
    import figstyle
    def _tidy_corner(fig):
        """Fewer, smaller tick labels so adjacent corner panels do not
        collide once the figure is printed at column width."""
        for ax in fig.axes:
            ax.tick_params(labelsize=9, pad=1.0)
            for which in ("x", "y"):
                ax.locator_params(axis=which, nbins=3)

    want = ["IRonly_free", "optical_free", "IR_CO053_cloud2.0", "IR_CO053_cloud2.5",
            "IR_CO053_cloud3.0", "IR_CO053_cloud4.5"]
    runs = {}
    for lab in want:
        d = load(lab)
        if d is None:
            print("missing:", lab)
            continue
        post = d["post"].copy()
        names = list(d["meta"]["stats"].keys())
        j = names.index("Rp")
        post[:, j] /= R_JUP
        d["post"] = post
        d["names"] = names
        runs[lab] = d
        print("loaded %-16s n=%d  params=%s" % (lab, post.shape[0], names))

    json.dump({k: {"n": int(v["post"].shape[0]),
                   "names": v["names"],
                   "stats": v["meta"]["stats"],
                   "ess": v["meta"].get("ess"),
                   "acceptance_fraction": v["meta"].get("acceptance_fraction"),
                   "correlations": v["meta"].get("correlations")}
               for k, v in runs.items()},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)

    print()
    for lab, d in runs.items():
        s = d["meta"]["stats"]
        print("%-16s n=%d acc=%.3f" % (lab, d["post"].shape[0],
                                       d["meta"].get("acceptance_fraction", -1)))
        for nm in d["names"]:
            x = s[nm]
            print("   %-15s %9.4f [%9.4f, %9.4f]  IQR=%.4f  prior_frac=%.3f  "
                  "frac_to_low_edge=%.3f"
                  % (nm, x["median"], x["lo68"], x["hi68"], x["iqr"],
                     x["iqr_frac"], x["dist_to_lower_edge"]))
        if "correlations" in d["meta"]:
            print("   corr:", {k: round(v, 3)
                               for k, v in d["meta"]["correlations"].items()})

    # ---- figure 8: coverage comparison ------------------------------------
    if "IRonly_free" in runs and "optical_free" in runs:
        n1 = runs["IRonly_free"]["names"]
        n2 = runs["optical_free"]["names"]
        common = [n for n in n1 if n in n2]
        i1 = [n1.index(n) for n in common]
        i2 = [n2.index(n) for n in common]
        labels = [LABEL[n] for n in common]
        fig = corner.corner(runs["IRonly_free"]["post"][:, i1], labels=labels,
                            color="0.35", plot_datapoints=False,
                            plot_contours=True, levels=[0.68, 0.95],
                            contour_kwargs=dict(linewidths=[1.1, 0.6]),
                            label_kwargs=dict(fontsize=14), show_titles=False)
        corner.corner(runs["optical_free"]["post"][:, i2], labels=labels, fig=fig,
                      color="tab:cyan", plot_datapoints=False, plot_contours=True,
                      levels=[0.68, 0.95], contour_kwargs=dict(linewidths=[1.1, 0.6]),
                      show_titles=False)
        h = [matplotlib.lines.Line2D([], [], color="0.35", lw=1.5,
                                     label="WASP-39\\,b G395H, IR-only"),
             matplotlib.lines.Line2D([], [], color="tab:cyan", lw=1.5,
                                     label="WASP-39\\,b NIRISS, 0.6--2.8 $\\mu$m")]
        _tidy_corner(fig)
        fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 0.995),
                   ncol=1, fontsize=11.0, frameon=True, framealpha=0.9,
                   edgecolor="0.7", handlelength=1.4, labelspacing=0.4)
        _enforce_fonts(fig)
        fig.set_size_inches(11.5, 11.5)
        fig.savefig(os.path.join(FIG, "fig8_corner_coverage.pdf"), bbox_inches="tight")
        print("wrote fig8_corner_coverage.pdf")

    # ---- figure 9: frozen-cloud comparison --------------------------------
    if "IR_CO053_cloud2.0" in runs and "IR_CO053_cloud4.5" in runs:
        n3 = runs["IR_CO053_cloud2.0"]["names"]
        labels = [LABEL[n] for n in n3]
        fig = corner.corner(runs["IR_CO053_cloud2.0"]["post"], labels=labels,
                            color="tab:orange", plot_datapoints=False,
                            plot_contours=True, levels=[0.68, 0.95],
                            contour_kwargs=dict(linewidths=[1.1, 0.6]),
                            label_kwargs=dict(fontsize=14), show_titles=False)
        corner.corner(runs["IR_CO053_cloud4.5"]["post"], labels=labels, fig=fig,
                      color="tab:purple", plot_datapoints=False,
                      plot_contours=True, levels=[0.68, 0.95],
                      contour_kwargs=dict(linewidths=[1.1, 0.6]),
                      show_titles=False)
        h = [matplotlib.lines.Line2D([], [], color="tab:orange", lw=1.5,
                                     label=r"cloud fixed at $\log P_{\rm c}=2.0$"),
             matplotlib.lines.Line2D([], [], color="tab:purple", lw=1.5,
                                     label=r"cloud fixed at $\log P_{\rm c}=4.5$")]
        # legend BELOW the corner grid: "upper right" sat on top of the 1-D panels
        _tidy_corner(fig)
        fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.99, 0.99),
                   ncol=1, fontsize=12.0, frameon=True, framealpha=0.9,
                   edgecolor="0.7", handlelength=1.4, labelspacing=0.4)
        _enforce_fonts(fig)
        fig.set_size_inches(7.0, 7.4)
        fig.savefig(os.path.join(FIG, "fig9_corner_cloudfixed.pdf"), bbox_inches="tight")
        print("wrote fig9_corner_cloudfixed.pdf")


if __name__ == "__main__":
    main()
