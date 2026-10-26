"""stage21_coverage_control.py
=============================
A CONTROLLED coverage test, requested by the editor: hold the planet, the
reduction, the noise model, the fiducial state and the priors fixed, and vary
ONLY the wavelength coverage by discarding channels from one real spectrum.

Previous evidence compared different spectra (different planets, instruments
and reductions), so coverage and precision were confounded. Here we take the
WASP-39 b NIRSpec PRISM spectrum (0.53-5.34 um, 207 channels, the widest
coverage in the corpus) and recompute
    d_s = tr A   and   the cloud-opacity horizon (L_logPc = 0.5)
for four progressively redder cuts of the SAME spectrum:

    full (0.53-5.34 um) | >1 um | >2 um | >3 um

The per-channel errors are kept as published, so cutting blue channels also
changes n_ch and the median error; we report both, which is the point: if d_s
and the horizon track the cut while the median error does not degrade, coverage
-- not precision -- is what moves the diagnostic.

Real data only; nothing is simulated.  Output: stage21_coverage.json
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage2_real_diagnostics import load_systems, load_datasets, PRIOR_STD
from stage4_horizon_and_bias import make_fwd, leverage_cloud, prior_cov
import identifiability as idf

OUT = os.path.join(HERE, "stage21_coverage.json")
CK = os.path.join(HERE, "ck_stage21.json")

PLANET = "WASP-39 b"
REFKEY = "RUSTAMKULOV_ET_AL__202"
NCH = 207
CUTS = [(0.0, "full 0.53--5.34 um"), (1.0, ">1 um"), (2.0, ">2 um"), (3.0, ">3 um")]
LPC_FID = 4.0


def pick(datasets):
    for ds in datasets:
        if ds["planet"] == PLANET and ds["ref"][:16] == REFKEY[:16] and len(ds["wl_um"]) == NCH:
            return ds
    raise SystemExit("spectrum not found")


def main():
    systems, datasets = load_systems(), load_datasets()
    ds0 = pick(datasets)
    sysp = systems[PLANET]
    Tfid = float(sysp["Teq"])
    cache = json.load(open(CK, encoding="utf-8")) if os.path.exists(CK) else {}
    out = []

    for wlmin, tag in CUTS:
        keep = ds0["wl_um"] >= wlmin
        ds = dict(ds0)
        ds["wl_um"] = ds0["wl_um"][keep]
        ds["sigma"] = ds0["sigma"][keep]
        fwd = make_fwd(sysp, ds)

        key_d = "%s|%.1f|dofs" % (REFKEY, wlmin)
        if key_d in cache:
            dsim = cache[key_d]
        else:
            L, F, J, th0 = leverage_cloud(fwd, sysp, ds, LPC_FID, Tfid)
            dsim = float(idf.dofs(F))
            cache[key_d] = dsim
            json.dump(cache, open(CK, "w", encoding="utf-8"))

        def L_of(x):
            k = "%s|%.1f|L|%.4f" % (REFKEY, wlmin, float(x))
            if k in cache:
                return cache[k]
            v = leverage_cloud(fwd, sysp, ds, float(x), Tfid)[0]
            cache[k] = v
            json.dump(cache, open(CK, "w", encoding="utf-8"))
            return v

        t0 = time.time()
        lo, hi = 2.0, 6.0
        horizon = None
        if L_of(lo) < 0.5 < L_of(hi):
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if L_of(mid) < 0.5:
                    lo = mid
                else:
                    hi = mid
            horizon = round(0.5 * (lo + hi), 3)

        row = dict(cut=tag, wl_min=float(wlmin),
                   wl_range=[float(ds["wl_um"].min()), float(ds["wl_um"].max())],
                   n_channels=int(len(ds["wl_um"])),
                   sigma_median_ppm=float(np.median(ds["sigma"]) * 1e4),
                   dofs=round(dsim, 3), horizon_logPc=horizon)
        out.append(row)
        print(json.dumps(row), "  (%.1fs)" % (time.time() - t0), flush=True)

    # ---- precision-matched control: take the reddest cut and rescale its
    # per-channel errors so its MEDIAN equals that of the full spectrum. If the
    # loss of d_s and of horizon depth were a precision effect, this rescale
    # would restore them; if it is a coverage effect, nothing changes.
    ref_med = float(np.median(ds0["sigma"]))
    keep = ds0["wl_um"] >= 3.0
    ds = dict(ds0)
    ds["wl_um"] = ds0["wl_um"][keep]
    sig = ds0["sigma"][keep] * (ref_med / float(np.median(ds0["sigma"][keep])))
    ds["sigma"] = sig
    fwd = make_fwd(sysp, ds)
    L, F, J, th0 = leverage_cloud(fwd, sysp, ds, LPC_FID, Tfid)
    dsig = float(idf.dofs(F))

    def L2(x):
        return leverage_cloud(fwd, sysp, ds, float(x), Tfid)[0]

    lo, hi = 2.0, 6.0
    horizon = None
    if L2(lo) < 0.5 < L2(hi):
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if L2(mid) < 0.5:
                lo = mid
            else:
                hi = mid
        horizon = round(0.5 * (lo + hi), 3)
    out.append(dict(cut=">3 um, errors rescaled to the full-spectrum median",
                    wl_min=3.0, wl_range=[float(ds["wl_um"].min()), float(ds["wl_um"].max())],
                    n_channels=int(len(ds["wl_um"])),
                    sigma_median_ppm=float(np.median(ds["sigma"]) * 1e6),
                    dofs=round(dsig, 3), horizon_logPc=horizon))
    # fix the unit label for the first four rows (sigma is a depth fraction)
    for r in out:
        r["sigma_median_ppm"] = round(r["sigma_median_ppm"] * 100.0, 1)
    print(json.dumps(out[-1]), flush=True)

    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
