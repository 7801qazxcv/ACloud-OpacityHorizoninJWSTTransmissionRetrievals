"""
stage17_prior_robustness.py  --  back the claim made in the appendix/limitations
===============================================================================
Claim under test: "multiplying every prior width by 0.5 or 2 shifts d_s by
<0.15 and leaves the horizon unchanged to three decimals".

Part 1 uses the Jacobians already computed for the twelve real spectra
(ck_stage2/*.npz contains J, the fiducial state, the real per-channel sigma),
so no forward model is re-run: d_s is recomputed with S_a scaled by 0.25 and 4
in variance (i.e. sigma x 0.5 and x 2).

Part 2 recomputes the horizon itself for one IR-only data set with the scaled
priors, by bisection, so that the second half of the claim is also backed.
"""
import json, os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_jup
from stage2_real_diagnostics import (load_systems, load_datasets,
                                     bins_from_centres, PNAMES, FD, PRIOR_STD)
import identifiability as idf

OUT = os.path.join(HERE, "stage17_prior.json")


def dofs_for(J, sigma, pstd):
    S_a = np.asarray(pstd, float) ** 2
    F = idf.fisher(idf.whiten_jacobian(J, sigma, S_a))
    return float(idf.resolution_matrix_summary(F, PNAMES)["dofs"]), idf.prior_leverage(F)


def part1():
    files = sorted(glob.glob(os.path.join(HERE, "ck_stage2", "*.npz")))
    rows = []
    for f in files:
        z = np.load(f)
        J, sigma, th0 = z["J"], z["sigma"], z["theta0"]
        pstd = np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                         PRIOR_STD["log_cloudtop_P"], 0.01 * th0[4]])
        base, Lb = dofs_for(J, sigma, pstd)
        lo, Llo = dofs_for(J, sigma, pstd * 0.5)
        hi, Lhi = dofs_for(J, sigma, pstd * 2.0)
        rows.append(dict(file=os.path.basename(f), n=int(len(sigma)),
                         dofs_base=round(base, 4), dofs_x0p5=round(lo, 4),
                         dofs_x2=round(hi, 4),
                         d_0p5=round(lo - base, 4), d_2=round(hi - base, 4),
                         L_cloud_base=round(float(Lb[3]), 5)))
        print("  %-52s n=%4d  d_s=%.3f  x0.5:%.3f (%+.3f)  x2:%.3f (%+.3f)"
              % (os.path.basename(f)[:52], len(sigma), base, lo, lo - base, hi, hi - base))
    d = np.array([[r["d_0p5"], r["d_2"]] for r in rows])
    print("  max |shift| over all 12 sets and both scalings: %.3f" % np.abs(d).max())
    return rows


def part2(planet="WASP-39", ref="ALDERSON"):
    """Horizon with the scaled priors, one IR-only spectrum, by bisection."""
    sy = load_systems()["WASP-39 b"]
    ds = [d for d in load_datasets() if planet in d["planet"] and ref in d["ref"].upper()][0]
    calc = TransitDepthCalculator()
    calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))
    sigma = ds["sigma"]
    Tfid, Rp0 = float(sy["Teq"]), float(sy["Rp"]) / R_jup
    steps = np.array([FD["T"], FD["logZ"], FD["CO_ratio"], FD["log_cloudtop_P"], 0.002])

    def fwd(th):
        _, dd, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[4] * R_jup, th[0],
                                       logZ=th[1], CO_ratio=th[2],
                                       cloudtop_pressure=10.0 ** th[3])
        return np.asarray(dd, float)

    def L_of(scale, lp):
        th = np.array([Tfid, 1.0, 0.53, float(lp), Rp0])
        f0 = fwd(th)
        J = np.zeros((f0.size, 5))
        for j in range(5):
            tp = th.copy(); tp[j] += steps[j]
            tm = th.copy(); tm[j] -= steps[j]
            J[:, j] = (fwd(tp) - fwd(tm)) / (2 * steps[j])
        pstd = np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                         PRIOR_STD["log_cloudtop_P"], 0.01 * Rp0]) * scale
        F = idf.fisher(idf.whiten_jacobian(J, sigma, np.asarray(pstd, float) ** 2))
        return float(idf.prior_leverage(F)[3])

    out = {}
    for label, scale in (("base", 1.0), ("x0.5", 0.5), ("x2", 2.0)):
        lo, hi = 3.0, 4.0
        flo, fhi = L_of(scale, lo), L_of(scale, hi)
        assert (flo - 0.5) * (fhi - 0.5) < 0, (label, flo, fhi)
        for i in range(20):
            mid = 0.5 * (lo + hi)
            fm = L_of(scale, mid)
            if (fm - 0.5) * (flo - 0.5) <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        out[label] = round(float(0.5 * (lo + hi)), 4)
        print("  horizon (%s) = %.4f" % (label, out[label]))
    print("  horizon shift: x0.5 %+.4f dex, x2 %+.4f dex"
          % (out["x0.5"] - out["base"], out["x2"] - out["base"]))
    return out


if __name__ == "__main__":
    rows = part1()
    h = part2()
    json.dump(dict(dofs_rows=rows, horizon=h), open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)
