"""
stage13_chemistry.py  --  REVISION ROUND 1 (referee #17, second half)
=====================================================================
Referee comment 17 asks whether the claim that the metallicity is
prior-dominated (and the location of the horizon) depends on WHICH molecules
the model contains.  This is testable directly: PLATON accepts a
`zero_opacities` argument, so we can recompute the same diagnostic with
individual species removed, leaving every other ingredient identical (same
planet, same real wavelength grid, same real per-channel sigma, same prior).

Removing an absorber is a deliberately extreme version of what disequilibrium
chemistry does: photochemistry and vertical mixing change the relative
abundances of these same species, so the end members bracket the effect. We
remove the two species that dominate the 3-5 um window (H2O, CO2) and the
pair that matters most at low metallicity (CH4, CO), plus all four together.

Output: stage13_chemistry.json  (horizon + fiducial leverage per variant)
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_jup
from stage2_real_diagnostics import (load_systems, load_datasets,
                                     bins_from_centres, PNAMES, FD, PRIOR_STD)
import identifiability as idf

OUT = os.path.join(HERE, "stage13_chemistry.json")

VARIANTS = [
    ("baseline", []),
    ("no_H2O", ["H2O"]),
    ("no_CO2", ["CO2"]),
    ("no_CH4_CO", ["CH4", "CO"]),
    ("no_H2O_CO2", ["H2O", "CO2"]),
    ("only_H2O", ["CO2", "CH4", "CO", "NH3", "HCN", "H2S", "SO2", "PH3",
                  "C2H2", "Na", "K", "TiO", "VO", "FeH"]),
]

PLANET, REF = "WASP-39", "ALDERSON"
LOGPC_COARSE = np.arange(1.5, 6.51, 0.25)


def get_dataset():
    for d in load_datasets():
        if PLANET in d["planet"] and REF in d["ref"].upper():
            return d
    raise KeyError(PLANET + " " + REF)


def make_fwd(calc, sy, zero):
    def fwd(th):
        _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[4] * R_jup, th[0],
                                      logZ=th[1], CO_ratio=th[2],
                                      cloudtop_pressure=10.0 ** th[3],
                                      zero_opacities=list(zero))
        return np.asarray(d, float)
    return fwd


def jac(fwd, th0, steps):
    try:
        f0 = fwd(th0)
        if not np.all(np.isfinite(f0)):
            return None
        J = np.zeros((f0.size, len(th0)))
        for j in range(len(th0)):
            tp = th0.copy(); tp[j] += steps[j]
            tm = th0.copy(); tm[j] -= steps[j]
            fp, fm = fwd(tp), fwd(tm)
            if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fm))):
                return None
            J[:, j] = (fp - fm) / (2 * steps[j])
        return J
    except Exception:
        return None


def leverage_cloud(J, sigma, S_a):
    F = idf.fisher(idf.whiten_jacobian(J, sigma, S_a))
    return float(idf.prior_leverage(F)[3])


def refine(fwd, sigma, S_a, th_base, steps, lo, hi, iters=18):
    """Bisect for L=0.5 in [lo, hi]; returns (logPc_crit, success)."""
    def L_at(lp):
        th = th_base.copy(); th[3] = float(lp)
        J = jac(fwd, th, steps)
        if J is None:
            return None
        return leverage_cloud(J, sigma, S_a)
    flo = L_at(lo)
    fhi = L_at(hi)
    if flo is None or fhi is None:
        return None, False
    # we want the crossing from L<0.5 (low logPc, cloud high) to L>0.5
    if (flo - 0.5) * (fhi - 0.5) > 0:
        return None, False
    for i in range(iters):
        mid = 0.5 * (lo + hi)
        fm = L_at(mid)
        if fm is None:
            return None, False
        if (fm - 0.5) * (flo - 0.5) <= 0:
            hi = mid; fhi = fm
        else:
            lo = mid; flo = fm
    return float(0.5 * (lo + hi)), True


def main():
    sy = load_systems()["WASP-39 b"]
    ds = get_dataset()
    sigma = ds["sigma"]
    Tfid = float(sy["Teq"])
    Rp0 = float(sy["Rp"]) / R_jup
    steps = np.array([FD["T"], FD["logZ"], FD["CO_ratio"], FD["log_cloudtop_P"], 0.002])
    pstd = np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                     PRIOR_STD["log_cloudtop_P"], 0.01 * Rp0])
    S_a = pstd ** 2
    rows = []
    for name, zero in VARIANTS:
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))
        fwd = make_fwd(calc, sy, zero)
        th_base = np.array([Tfid, 1.0, 0.53, 4.0, Rp0])
        # coarse scan for the bracket
        vals = []
        for lpc in LOGPC_COARSE:
            th = th_base.copy(); th[3] = float(lpc)
            J = jac(fwd, th, steps)
            vals.append(leverage_cloud(J, sigma, S_a) if J is not None else np.nan)
        hv = np.array(vals)
        ok = np.isfinite(hv)
        horizon, okk = None, False
        sgn = (hv[:-1] - 0.5) * (hv[1:] - 0.5)
        if np.any(ok[:-1] & ok[1:] & (sgn <= 0)):
            k = int(np.where(ok[:-1] & ok[1:] & (sgn <= 0))[0][0])
            horizon, okk = refine(fwd, sigma, S_a, th_base, steps,
                                  LOGPC_COARSE[k], LOGPC_COARSE[k + 1])
        # leverage at the standard fiducial logPc = 4
        th4 = th_base.copy(); th4[3] = 4.0
        J4 = jac(fwd, th4, steps)
        L4 = leverage_cloud(J4, sigma, S_a) if J4 is not None else None
        rows.append(dict(variant=name, zero_opacities=zero,
                         horizon_logPc=horizon if okk else None,
                         L_cloud_fiducial=L4,
                         L_curve=[None if not np.isfinite(v) else float(v) for v in hv]))
        print("  %-12s horizon=%s  L_logPc(4dex)=%s"
              % (name, ("%.3f" % horizon) if okk else "none",
                 ("%.3f" % L4) if L4 is not None else "n/a"), flush=True)
        json.dump(dict(dataset="WASP-39b Alderson G395H", rows=rows),
                  open(OUT, "w", encoding="utf-8"), indent=1)
    json.dump(dict(dataset="WASP-39b Alderson G395H", rows=rows),
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
