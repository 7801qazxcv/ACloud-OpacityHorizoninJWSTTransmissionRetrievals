"""
stage14_refine.py  --  refine the robustness horizons by bisection
==================================================================
stage11 located the L=0.5 crossing by linear interpolation on a 0.25-dex grid,
which is adequate for spotting trends but too coarse to quote. This script
re-locates each crossing by bisection to ~0.01 dex and writes everything the
manuscript needs, together with a figure:

  stage14_refine.json   refined horizons for
                        (a) the 14 fiducial atmospheres of stage11 Part A
                            (WASP-39 b G395H, varying T, logZ, C/O)
                        (b) the four metallicity cases of stage11 Part C
                            (GJ 486 b, logZ = 0,1,2,3 -> mean molecular weight)
  ../figures/fig7_horizon_vs_state.pdf
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(HERE, "stage14_refine.json")

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_jup
from stage2_real_diagnostics import (load_systems, load_datasets,
                                     bins_from_centres, FD, PRIOR_STD)
from stage11_robustness import jac, cloud_leverage, column_corr, pstd

STEPS5 = np.array([FD["T"], FD["logZ"], FD["CO_ratio"], FD["log_cloudtop_P"], 0.002])


def pick(planet, ref):
    cands = [d for d in load_datasets() if planet in d["planet"] and ref in d["ref"].upper()]
    return max(cands, key=lambda d: len(d["wl_um"]))


def maker(sy, ds):
    calc = TransitDepthCalculator()
    calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))
    sigma = ds["sigma"]

    def fwd(th):
        _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[4] * R_jup, th[0],
                                      logZ=th[1], CO_ratio=th[2],
                                      cloudtop_pressure=10.0 ** th[3])
        return np.asarray(d, float)

    return fwd, sigma


def L_of(fwd, sigma, S_a, th_base, lp, idx=3):
    th = np.array(th_base, float); th[3] = float(lp)
    J, _ = jac(fwd, th, fd=STEPS5)
    if J is None:
        return None
    return cloud_leverage(J, sigma, S_a, idx)[0]


def refine(fwd, sigma, S_a, th_base, h_guess, iters=18, half=0.25):
    lo, hi = h_guess - half, h_guess + half
    flo, fhi = L_of(fwd, sigma, S_a, th_base, lo), L_of(fwd, sigma, S_a, th_base, hi)
    tries = 0
    while (flo is None or fhi is None or (flo - 0.5) * (fhi - 0.5) > 0) and tries < 3:
        tries += 1; half *= 2
        lo, hi = h_guess - half, h_guess + half
        flo = L_of(fwd, sigma, S_a, th_base, lo)
        fhi = L_of(fwd, sigma, S_a, th_base, hi)
    if flo is None or fhi is None or (flo - 0.5) * (fhi - 0.5) > 0:
        return None, False, (None if flo is None else round(flo, 3)), (None if fhi is None else round(fhi, 3))
    for i in range(iters):
        mid = 0.5 * (lo + hi)
        fm = L_of(fwd, sigma, S_a, th_base, mid)
        if fm is None:
            break
        if (fm - 0.5) * (flo - 0.5) <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return float(0.5 * (lo + hi)), True, round(flo, 4), round(fhi, 4)


def main():
    out = {"fiducial": [], "gj486": []}

    # ---------------- (a) WASP-39 b fiducial grid -------------------------
    A = json.load(open(os.path.join(HERE, "stage11_fiducial.json"), encoding="utf-8"))
    sy = load_systems()["WASP-39 b"]
    ds = pick("WASP-39", "ALDERSON")
    fwd, sigma = maker(sy, ds)
    Rp0 = float(sy["Rp"]) / R_jup
    for r in A["per_fiducial"]:
        if r["horizon_logPc"] != r["horizon_logPc"]:
            out["fiducial"].append(dict(**r, refined=None, refined_ok=False))
            continue
        T, lz, co = r["fiducial"]["T"], r["fiducial"]["logZ"], r["fiducial"]["CO"]
        th_base = np.array([T, lz, co, 4.0, Rp0])
        S_a = pstd(th_base) ** 2
        h, okk, flo, fhi = refine(fwd, sigma, S_a, th_base, r["horizon_logPc"])
        # cross-term diagnostics at the standard fiducial cloud pressure
        th4 = th_base.copy(); th4[3] = 4.0
        J4, _ = jac(fwd, th4, fd=STEPS5)
        cors = None
        if J4 is not None:
            cors = dict(logZ_cloud=column_corr(J4, sigma, 1, 3),
                        CO_cloud=column_corr(J4, sigma, 2, 3),
                        Rp_cloud=column_corr(J4, sigma, 4, 3))
        out["fiducial"].append(dict(fiducial=r["fiducial"], coarse=r["horizon_logPc"],
                                    refined=None if h is None else round(h, 4),
                                    refined_ok=bool(okk), L_lo=flo, L_hi=fhi,
                                    column_corr_at_4dex=cors))
        print("  A T=%4.0f logZ=%.1f CO=%.2f  coarse=%.3f -> refined=%s"
              % (T, lz, co, r["horizon_logPc"], ("%.4f" % h) if okk else "none"), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    # ---------------- (b) GJ 486 b metallicity ----------------------------
    C = json.load(open(os.path.join(HERE, "stage11_mmw.json"), encoding="utf-8"))
    sy2 = load_systems()["GJ 486 b"]
    ds2 = pick("GJ 486", "MORAN")
    fwd2, sigma2 = maker(sy2, ds2)
    Rp02 = float(sy2["Rp"]) / R_jup
    for r in C["rows"]:
        if r["horizon_logPc"] != r["horizon_logPc"]:
            out["gj486"].append(dict(logZ=r["logZ"], mu_mean_amu=r["mu_mean_amu"],
                                     horizon_refined=None, note="all forward models failed"))
            print("  C logZ=%.1f  none (model does not converge)" % r["logZ"], flush=True)
            continue
        th_base = np.array([float(sy2["Teq"]), r["logZ"], 0.53, 4.0, Rp02])
        S_a = pstd(th_base) ** 2
        h, okk, flo, fhi = refine(fwd2, sigma2, S_a, th_base, r["horizon_logPc"])
        out["gj486"].append(dict(logZ=r["logZ"], mu_mean_amu=r["mu_mean_amu"],
                                 coarse=r["horizon_logPc"],
                                 horizon_refined=None if h is None else round(h, 4)))
        print("  C logZ=%.1f mu=%.2f coarse=%.3f -> refined=%s"
              % (r["logZ"], r["mu_mean_amu"], r["horizon_logPc"],
                 ("%.4f" % h) if okk else "none"), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    vals = [r["refined"] for r in out["fiducial"] if r.get("refined") is not None]
    if vals:
        print("fiducial spread: %.3f - %.3f  (n=%d)" % (min(vals), max(vals), len(vals)), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
