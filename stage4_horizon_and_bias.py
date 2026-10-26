"""
stage4_horizon_and_bias.py
==========================
Two experiments that turn the stage3 finding into the paper's core result.

A. CLOUD OPACITY HORIZON.  Locate P_c^crit for every dataset in the corpus by
   bisection on the leverage curve L_cloud(log Pc) = 0.5, using each dataset's
   real per-channel sigma. Tests whether the horizon is a universal property
   of transmission spectroscopy or depends on wavelength coverage / precision.

B. BIAS TRANSFER ACROSS THE HORIZON.  The practical consequence: when a cloud
   parameter is unidentifiable, retrievals that *fix* it (or let a prior pin it)
   transfer the error into metallicity. We quantify d(logZ)/d(logPc) exactly,
   via the linear bias-transfer operator, on both sides of the horizon.

   This is the "cost side" of the identifiability/bias trade-off that the paper
   argues must be reported alongside any physics-informed prior.

Outputs: stage4_results.json
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_jup
import identifiability as idf
from stage2_real_diagnostics import (load_systems, load_datasets, bins_from_centres,
                                     PNAMES, FD, PRIOR_STD)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage4_results.json")
CK = os.path.join(HERE, "ck_stage4"); os.makedirs(CK, exist_ok=True)


def make_fwd(sysp, ds):
    calc = TransitDepthCalculator()
    calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))

    def fwd(th):
        _, d, _ = calc.compute_depths(
            sysp["Rs"], sysp["Mp"], th[4] * R_jup, th[0],
            logZ=th[1], CO_ratio=th[2],
            cloudtop_pressure=10.0 ** th[3], full_output=False)
        return np.asarray(d, float)
    return fwd


def jac(fwd, th0):
    steps = np.array([FD[n] for n in PNAMES])
    J = np.zeros((fwd(th0).size, 5))
    for i in range(5):
        tp = th0.copy(); tp[i] += steps[i]
        tm = th0.copy(); tm[i] -= steps[i]
        J[:, i] = (fwd(tp) - fwd(tm)) / (2 * steps[i])
    return J


def prior_cov(sysp):
    return np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                     PRIOR_STD["log_cloudtop_P"], 0.01 * sysp["Rp"] / R_jup]) ** 2


def leverage_cloud(fwd, sysp, ds, lpc, Tfid):
    th0 = np.array([Tfid, 1.0, 0.53, float(lpc), sysp["Rp"] / R_jup])
    J = jac(fwd, th0)
    F = idf.fisher(idf.whiten_jacobian(J, ds["sigma"], prior_cov(sysp)))
    A = idf.averaging_kernel(F)
    return float(1 - A[3, 3]), F, J, th0


def main():
    systems, datasets = load_systems(), load_datasets()
    out = []
    for k, ds in enumerate(datasets):
        pl = ds["planet"]
        if pl not in systems:
            continue
        uid = "%s_%s_n%d" % (pl.replace(" ", ""), ds["ref"][:16], len(ds["wl_um"]))
        ckf = os.path.join(CK, uid + ".json")
        if os.path.exists(ckf):
            out.append(json.load(open(ckf)))
            print("[%2d] cached %s" % (k, uid), flush=True); continue

        sysp = systems[pl]
        Tfid = float(sysp["Teq"])
        fwd = make_fwd(sysp, ds)
        t0 = time.time()

        # ---- A. bisect for the horizon: L_cloud crosses 0.5 ----
        lo, hi = 2.0, 6.0
        Llo, _, _, _ = leverage_cloud(fwd, sysp, ds, lo, Tfid)
        Lhi, _, _, _ = leverage_cloud(fwd, sysp, ds, hi, Tfid)
        horizon = None
        if Llo < 0.5 < Lhi:
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                Lm, _, _, _ = leverage_cloud(fwd, sysp, ds, mid, Tfid)
                if Lm < 0.5:
                    lo, Llo = mid, Lm
                else:
                    hi, Lhi = mid, Lm
            horizon = 0.5 * (lo + hi)

        # ---- B. bias transfer, evaluated INSIDE the horizon (cloud detectable)
        # and BEYOND it (cloud unidentifiable). delta = 1 dex error in log Pc.
        rows = []
        for tag, lpc in (("inside", 2.5), ("beyond", 5.0)):
            L, F, J, th0 = leverage_cloud(fwd, sysp, ds, lpc, Tfid)
            free, bias = idf.bias_from_fixing(J, ds["sigma"], prior_cov(sysp),
                                              idx_fixed=[3], delta=[1.0])
            Sp = idf.posterior_covariance(F, prior_cov(sysp))
            sd = np.sqrt(np.diag(Sp))
            bmap = {PNAMES[j]: float(b) for j, b in zip(free, bias)}
            smap = {PNAMES[j]: float(sd[j]) for j in range(5)}
            rows.append(dict(regime=tag, log_cloudtop_P=lpc, leverage_cloud=L,
                             dofs=float(idf.dofs(F)),
                             bias_per_dex=bmap, posterior_sd=smap,
                             bias_in_sigma={n: (bmap[n] / smap[n] if n in bmap and smap[n] > 0 else None)
                                            for n in PNAMES}))

        rec = dict(planet=pl, instrument=ds["instrument"], ref=ds["ref"],
                   n_channels=int(len(ds["wl_um"])),
                   wl_um_min=float(ds["wl_um"].min()), wl_um_max=float(ds["wl_um"].max()),
                   has_optical=bool(ds["wl_um"].min() < 1.0),
                   sigma_ppm_median=float(np.median(ds["sigma"]) * 1e6),
                   Teq=Tfid, horizon_logPc=horizon, regimes=rows,
                   seconds=round(time.time() - t0, 1))
        json.dump(rec, open(ckf, "w"), indent=1)
        out.append(rec)
        print("[%2d] %-11s %-18s opt=%d sig=%6.1f horizon=%s  bias(logZ)/dex: in=%+.3f beyond=%+.3f  %.0fs"
              % (k, pl, ds["ref"][:18], rec["has_optical"], rec["sigma_ppm_median"],
                 ("%.2f" % horizon) if horizon else "none",
                 rows[0]["bias_per_dex"].get("logZ", float("nan")),
                 rows[1]["bias_per_dex"].get("logZ", float("nan")), rec["seconds"]), flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

    json.dump(out, open(OUT, "w"), indent=1)
    print("\nwrote", OUT, len(out), "datasets")
    hs = [r["horizon_logPc"] for r in out if r["horizon_logPc"]]
    if hs:
        print("horizon log10(Pc/Pa): mean=%.3f sd=%.3f min=%.3f max=%.3f  (n=%d)"
              % (np.mean(hs), np.std(hs), np.min(hs), np.max(hs), len(hs)))


if __name__ == "__main__":
    main()
