"""stage20_threshold.py
=====================
Reviewer-proofing experiment behind the paper's Figure/Threshold paragraph:

    Is the cloud-opacity horizon a physical transition, or merely the crossing
    point of an arbitrarily chosen diagnostic threshold L = 0.5?

For three representative spectra (an infrared-only gas giant, an optical+NIR
gas giant, and an infrared-only super-Earth) we recompute the prior-leverage
curve L_cloud(log Pc) on the SAME forward model, noise vector and prior used in
Stage 4, and locate the crossing for THREE thresholds,

    L_thr = 0.25, 0.50, 0.75,

by bisection between bracketing grid points. We also record whether the curve
is monotone in the search interval, because the leverage is NOT monotone at the
low-pressure end (the cloud becomes optically thin again there) and a threshold
set too low could produce a spurious second crossing.

Everything here is recomputed from the same public spectra; nothing is
simulated.  Output: stage20_threshold.json
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stage2_real_diagnostics import load_systems, load_datasets, bins_from_centres
from stage4_horizon_and_bias import (make_fwd, leverage_cloud, prior_cov)
import identifiability as idf

OUT = os.path.join(HERE, "stage20_threshold.json")
CK = os.path.join(HERE, "ck_stage20.json")

# representative data sets: (planet, first 16 chars of ref, n channels)
TARGETS = [
    ("WASP-39 b", "ALDERSON_ET_AL__2022", 344),    # IR-only hot Saturn      (G395H)
    ("WASP-39 b", "RUSTAMKULOV_ET_AL__202", 207),  # optical+NIR hot Saturn  (PRISM)
    ("GJ 486 b",  "MORAN_ET_AL__2023", 110),       # IR-only super-Earth     (G395H)
]
THRESHOLDS = [0.25, 0.50, 0.75]
GRID = np.arange(2.0, 6.001, 0.25)


def pick(datasets, planet, refkey, nch):
    for ds in datasets:
        if ds["planet"] == planet and ds["ref"][:16] == refkey[:16] \
                and len(ds["wl_um"]) == nch:
            return ds
    return None


def load_cache():
    return json.load(open(CK, encoding="utf-8")) if os.path.exists(CK) else {}


def main():
    systems, datasets = load_systems(), load_datasets()
    cache = load_cache()
    out = []
    for planet, refkey, nch in TARGETS:
        ds = pick(datasets, planet, refkey, nch)
        if ds is None:
            print("MISSING", planet, refkey, nch, flush=True)
            continue
        sysp = systems[planet]
        Tfid = float(sysp["Teq"])
        fwd = make_fwd(sysp, ds)

        def L_of(lpc):
            key = "%s|%s|%d|%.4f" % (planet, refkey, nch, float(lpc))
            if key in cache:
                return cache[key]
            v = leverage_cloud(fwd, sysp, ds, float(lpc), Tfid)[0]
            cache[key] = v
            json.dump(cache, open(CK, "w", encoding="utf-8"))
            return v

        curve = []
        for x in GRID:
            t0 = time.time()
            curve.append(float(L_of(x)))
            print("  %s %s  logPc=%.2f  L=%.4f  (%.1fs)"
                  % (planet, refkey, x, curve[-1], time.time() - t0), flush=True)

        res = dict(planet=planet, ref=refkey, n_channels=nch,
                   instrument=ds.get("instrument", ""),
                   has_optical=bool(float(np.min(ds["wl_um"])) < 1.0),
                   grid=[float(x) for x in GRID], curve=curve)
        # bisect for each threshold between bracketing grid points
        for thr in THRESHOLDS:
            xlo, xhi, found = None, None, False
            for i in range(len(GRID) - 1):
                a, b = curve[i], curve[i + 1]
                if a < thr <= b:
                    xlo, xhi = float(GRID[i]), float(GRID[i + 1])
                    found = True
                    break
            if not found:
                res["Pcrit_L%.2f" % thr] = None
                continue
            for _ in range(10):
                mid = 0.5 * (xlo + xhi)
                if L_of(mid) < thr:
                    xlo = mid
                else:
                    xhi = mid
            res["Pcrit_L%.2f" % thr] = round(0.5 * (xlo + xhi), 3)
        # monotonicity check over the search window
        c = np.array(curve)
        seg = c[np.asarray(GRID) >= 3.0]
        res["monotone_above_3"] = bool(np.all(np.diff(seg) >= -0.02))
        res["L_min_below_3"] = float(np.min(c[np.asarray(GRID) <= 3.0]))
        out.append(res)
        print(json.dumps({k: v for k, v in res.items()
                          if k not in ("grid", "curve")}), flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
