"""
stage9_uncertainty.py  --  REVISION ROUND 1 (referee item #12 and #13)
======================================================================
Referee #12 asks whether S_epsilon is built from the published per-channel
uncertainties or from a single median value per data set, and asks for the
channel-to-channel variation to be SHOWN rather than asserted.

Answer (measured here, from the same archive columns used everywhere else in
the paper):

  * the diagnostic uses the per-channel sigma_i, NOT the median (see
    identifiability.py: `sigma` is a length-n vector and K is whitened as
    K / sigma[:,None]);
  * the per-channel sigma varies strongly within a data set, and the variation
    is structured (detector edges / order joins), which is why a single median
    is not an adequate description.

This script measures that structure on the twelve real spectra and writes
stage9_uncertainty.json + ../figures/fig6_uncertainty.pdf.  Nothing simulated.
"""
import csv, collections, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figures")
OUT = os.path.join(HERE, "stage9_uncertainty.json")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "savefig.dpi": 300,
    "axes.labelsize": 9, "legend.fontsize": 7.0, "legend.frameon": False,
})
CW, DW = 3.5, 7.2


# Shared house style: minimal labels, one colour family per planet,
# no top/right spines, faint dashed grid, unified font size.
import figstyle
figstyle.unify_fonts()


def load_spectra():
    rows = list(csv.DictReader(open(os.path.join(HERE, "transitspec_jwst.csv"),
                                    encoding="utf-8")))
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
                wl.append(float(r["centralwavelng"]))
                dep.append(float(r["plntransdep"]))
                err.append(abs(float(r["plntransdeperr1"])))
            except (TypeError, ValueError):
                continue
        if len(wl) < 8:
            continue
        o = np.argsort(wl)
        out.append(dict(planet=pl, instrument=inst, ref=tag,
                        wl=np.array(wl)[o], dep=np.array(dep)[o],
                        err=np.array(err)[o]))
    return out


def mode_key(d):
    inst = d["instrument"].upper()
    if "NIRISS" in inst:
        return "NIRISS SOSS"
    if "NIRCAM" in inst:
        return "NIRCam"
    if "NIRSPEC" in inst:
        # The Archive records the grating for some data sets and the reduction
        # pipeline (Tiberius / FIREFLy / Eureka! / tshirt) for others; the
        # shortest observed wavelength separates PRISM from G395H regardless.
        return "NIRSpec PRISM" if float(np.min(d["wl"])) < 1.0 else "NIRSpec G395H"
    return "other"


def short(tag):
    s = tag.replace("_ET_AL__", "+").replace(
        "JWST_TRANSITING_EXOPLANET_COMMUNITY_EARLY_RELEASE_SCIENCE_TEAM", "JTEC ERS")
    return s.replace("__", " ").rstrip("_")


def main():
    sp = load_spectra()
    sp.sort(key=lambda d: (mode_key(d), d["planet"], d["wl"].min()))
    rows = []
    for d in sp:
        s = d["err"] * 1e4                      # percent -> ppm
        q10, q50, q90 = np.percentile(s, [10, 50, 90])
        n = len(s)
        edge = np.concatenate([s[:max(n // 10, 1)], s[-max(n // 10, 1):]])
        core = np.percentile(s, 50)
        rows.append(dict(
            planet=d["planet"], instrument=mode_key(d), ref=short(d["ref"])[:28],
            n_channels=int(n), wl_min=float(d["wl"].min()), wl_max=float(d["wl"].max()),
            sigma_p10=float(q10), sigma_median=float(q50), sigma_p90=float(q90),
            sigma_min=float(s.min()), sigma_max=float(s.max()),
            sigma_max_over_min=float(s.max() / s.min()),
            sigma_p90_over_p10=float(q90 / q10),
            edge_to_median=float(np.median(edge) / core)))
    json.dump(rows, open(OUT, "w", encoding="utf-8"), indent=1)

    modes = ["NIRISS SOSS", "NIRSpec PRISM", "NIRSpec G395H", "NIRCam"]
    fig, axes = plt.subplots(2, 2, figsize=(DW, 6.6))
    for ax, mode in zip(axes.ravel(), modes):
        ds = [d for d in sp if mode_key(d) == mode]
        figstyle.reset_counter()
        for d in ds:
            s = d["err"] * 1e4
            kw = figstyle.style_for(d["planet"], marker=True, alpha=0.75)
            ax.plot(d["wl"], s, label=figstyle.short_label(d["planet"], len(s)), **kw)
        # the median of each data set, drawn faintly under the curves
        for i, d in enumerate(ds):
            s = d["err"] * 1e4
            fam = figstyle.family_of(d["planet"])
            ax.plot([d["wl"].min(), d["wl"].max()], [np.median(s)] * 2,
                    ls=":", lw=1.0, color=fam[i % len(fam)], alpha=0.55)
        ax.set_yscale("log")
        allw = np.concatenate([d["wl"] for d in ds])
        alls = np.concatenate([d["err"] * 1e4 for d in ds])
        figstyle.tight_limits(ax, x=allw, y=alls, pad=0.03)
        figstyle.tidy_axes(ax)
        ax.set_xlabel(r"wavelength [$\mu$m]")
        ax.set_ylabel(r"$\sigma_i$ [ppm]")
        ax.set_title(mode, fontsize=figstyle.FONTSIZE, pad=6)
    # ONE shared legend at the bottom of the grid: three-column per-panel
    # legends collided inside the panel width.
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=figstyle.FONTSIZE - 3, frameon=False,
               handlelength=1.8, columnspacing=1.2, labelspacing=0.35)
    fig.suptitle("published per-channel uncertainties: structure, not a constant "
                 "(dotted = per-set median)", fontsize=figstyle.FONTSIZE, y=0.985)
    fig.subplots_adjust(hspace=0.42, top=0.91, bottom=0.17, wspace=0.24)
    figstyle.save(fig, os.path.join(FIG, "fig6_uncertainty.pdf"))
    print("wrote", os.path.join(FIG, "fig6_uncertainty.pdf"))
    print("n datasets", len(rows))
    for r in rows:
        print("  %-11s %-14s n=%4d  med=%6.1f ppm  p90/p10=%.2f  max/min=%.2f  edge/med=%.2f"
              % (r["planet"], r["instrument"], r["n_channels"], r["sigma_median"],
                 r["sigma_p90_over_p10"], r["sigma_max_over_min"], r["edge_to_median"]))


if __name__ == "__main__":
    main()
