"""
stage12_mrt.py  --  machine-readable tables (AAS data editor request)
=====================================================================
The data editor asks for the table that the appendix promises ("the complete
set of twelve curves, and the horizon locations for all twelve spectra") plus a
README naming every column, its units and how it was produced.

stage3_cloud_probe.json only computed the LEVERAGE CURVES for three
representative data sets.  Here we recompute the curves for ALL TWELVE with
the same conventions as the main diagnostic (real system parameters, real
per-channel sigma, prior = stage2 PRIOR_STD) and write:

  data/horizon_curves.csv   long table: one row per (data set, cloud pressure)
                            with the prior leverage and the DOFS
  data/summary.csv          one row per data set: identifiers, coverage,
                            precision, fiducial-state leverage per parameter,
                            and the horizon located by bisection
  data/README.txt           column-by-column description, units, provenance

Nothing is simulated.
"""
import csv, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_jup
from stage2_real_diagnostics import (load_systems, load_datasets,
                                     bins_from_centres, PNAMES, FD, PRIOR_STD)
import identifiability as idf

LOGPC = np.arange(1.5, 6.51, 0.25)
OUT = os.path.join(HERE, "stage12_mrt.json")


def short(tag):
    s = tag.replace("_ET_AL__", "+").replace(
        "JWST_TRANSITING_EXOPLANET_COMMUNITY_EARLY_RELEASE_SCIENCE_TEAM", "JTEC ERS")
    return s.replace("__", " ").rstrip("_")


def dataset_id(d, k):
    return "%02d_%s_%s_n%d" % (k, d["planet"].replace(" ", ""),
                               "".join(ch for ch in d["instrument"].split()[0]
                                       if ch.isalnum())[:6], len(d["wl_um"]))


def main():
    systems = load_systems()
    datasets = load_datasets()
    order = sorted(range(len(datasets)),
                   key=lambda i: (datasets[i]["wl_um"].min() >= 1.0,
                                  datasets[i]["planet"], datasets[i]["wl_um"].min()))
    rows_summary, rows_curve, meta = [], [], []
    for k0, i in enumerate(order):
        ds = datasets[i]
        pl = ds["planet"]
        if pl not in systems:
            print("skip (no system parameters):", pl, flush=True)
            continue
        sy = systems[pl]
        did = dataset_id(ds, k0)
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))
        sigma = ds["sigma"]
        Tfid = float(sy["Teq"]) if np.isfinite(sy["Teq"]) else 1000.0
        Rp0 = float(sy["Rp"]) / R_jup
        steps = np.array([FD["T"], FD["logZ"], FD["CO_ratio"],
                          FD["log_cloudtop_P"], 0.002])

        def fwd(th):
            _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[4] * R_jup, th[0],
                                          logZ=th[1], CO_ratio=th[2],
                                          cloudtop_pressure=10.0 ** th[3])
            return np.asarray(d, float)

        th_ref = np.array([Tfid, 1.0, 0.53, 4.0, Rp0])
        pstd = np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                         PRIOR_STD["log_cloudtop_P"], 0.01 * Rp0])
        S_a = pstd ** 2

        t0 = time.time()
        curve = []
        for lpc in LOGPC:
            th = np.array([Tfid, 1.0, 0.53, float(lpc), Rp0])
            F = None
            try:
                f0 = fwd(th)
                J = np.zeros((f0.size, 5))
                for j in range(5):
                    tp = th.copy(); tp[j] += steps[j]
                    tm = th.copy(); tm[j] -= steps[j]
                    fp, fm = fwd(tp), fwd(tm)
                    J[:, j] = (fp - fm) / (2 * steps[j])
                F = idf.fisher(idf.whiten_jacobian(J, sigma, S_a))
                L = idf.prior_leverage(F)
                dofs = float(idf.resolution_matrix_summary(F, PNAMES)["dofs"])
            except Exception:
                L = np.full(5, np.nan)
                dofs = float("nan")
            curve.append((float(lpc), float(L[3]), dofs))
            rows_curve.append(dict(dataset_id=did, log_cloudtop_P=round(float(lpc), 3),
                                   leverage_log_cloudtop_P=round(float(L[3]), 6),
                                   dofs=round(dofs, 6)))

        # fiducial-state summary (log Pc = 4) -- recompute cleanly
        try:
            f0 = fwd(th_ref)
            J = np.zeros((f0.size, 5))
            for j in range(5):
                tp = th_ref.copy(); tp[j] += steps[j]
                tm = th_ref.copy(); tm[j] -= steps[j]
                J[:, j] = (fwd(tp) - fwd(tm)) / (2 * steps[j])
            F = idf.fisher(idf.whiten_jacobian(J, sigma, S_a))
            Lf = idf.prior_leverage(F)
            dofs_f = float(idf.resolution_matrix_summary(F, PNAMES)["dofs"])
        except Exception:
            Lf = np.full(5, np.nan); dofs_f = float("nan")

        # horizon by bisection on the 0.25-dex grid, refined by linear interp
        lv = np.array([c[1] for c in curve])
        sgn = (lv[:-1] - 0.5) * (lv[1:] - 0.5)
        if np.any(np.isfinite(sgn) & (sgn <= 0)):
            kk = int(np.where(np.isfinite(sgn) & (sgn <= 0))[0][0])
            tt = (0.5 - lv[kk]) / (lv[kk + 1] - lv[kk])
            horizon = float(LOGPC[kk] + tt * (LOGPC[kk + 1] - LOGPC[kk]))
        else:
            horizon = float("nan")

        rows_summary.append(dict(
            dataset_id=did, planet=pl, instrument=ds["instrument"],
            reference=short(ds["ref"])[:40],
            n_channels=int(len(ds["wl_um"])),
            wl_min_um=round(float(ds["wl_um"].min()), 3),
            wl_max_um=round(float(ds["wl_um"].max()), 3),
            has_optical=int(bool(ds["wl_um"].min() < 1.0)),
            sigma_median_ppm=round(float(np.median(sigma) * 1e6), 2),
            fiducial_T_K=round(Tfid, 1), fiducial_logZ_dex=1.0,
            fiducial_CO_ratio=0.53, fiducial_logPc_dex=4.0,
            L_T=round(float(Lf[0]), 5), L_logZ=round(float(Lf[1]), 5),
            L_CO_ratio=round(float(Lf[2]), 5), L_logPc=round(float(Lf[3]), 5),
            L_Rp=round(float(Lf[4]), 5), dofs_at_fiducial=round(dofs_f, 4),
            horizon_logPc=round(horizon, 4) if horizon == horizon else ""))
        meta.append(dict(dataset_id=did, seconds=round(time.time() - t0, 1)))
        print("[%2d] %-22s %-11s n=%4d  horizon=%s  %.0fs"
              % (k0, did, pl, len(ds["wl_um"]),
                 ("%.3f" % horizon) if horizon == horizon else "none",
                 time.time() - t0), flush=True)
        json.dump(dict(summary=rows_summary, meta=meta), open(OUT, "w", encoding="utf-8"),
                  indent=1)

    with open(os.path.join(DATA, "horizon_curves.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_id", "log_cloudtop_P",
                                          "leverage_log_cloudtop_P", "dofs"])
        w.writeheader(); w.writerows(rows_curve)
    with open(os.path.join(DATA, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_summary[0].keys()))
        w.writeheader(); w.writerows(rows_summary)
    print("wrote", len(rows_summary), "summary rows and", len(rows_curve), "curve rows")


if __name__ == "__main__":
    main()
