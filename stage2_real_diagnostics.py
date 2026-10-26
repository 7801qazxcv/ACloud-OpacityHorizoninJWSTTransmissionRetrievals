"""
stage2_real_diagnostics.py
==========================
Compute the identifiability diagnostic on EVERY published JWST transmission
spectrum in the corpus, using each dataset's OWN published per-channel
uncertainties. No synthetic noise, no assumed error floor.

For each dataset:
  1. read the real wavelength grid + real per-channel sigma from the archive
  2. build the PLATON forward model on exactly that grid
  3. finite-difference the Jacobian at a fiducial state
  4. report DOFS, prior leverage per parameter, degeneracy directions

Outputs: stage2_diagnostics.json  (+ per-dataset .npz with Jacobians)

INTEGRITY: the fiducial state is a MODEL evaluation point, not a claim about
the planet. The diagnostic (DOFS, leverage) is a property of the
measurement + model pair, which is exactly what the paper is about.
"""
import csv, collections, json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup, R_earth, M_earth
import identifiability as idf

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "transitspec_jwst.csv")
PARS = os.path.join(HERE, "system_params.csv")
OUT = os.path.join(HERE, "stage2_diagnostics.json")
CKDIR = os.path.join(HERE, "ck_stage2")
os.makedirs(CKDIR, exist_ok=True)

PNAMES = ["T", "logZ", "CO_ratio", "log_cloudtop_P", "Rp"]
# Finite-difference steps AND prior widths (see paper Table: prior specification)
# Prior std: T 150 K, logZ 1.0 dex, C/O 0.3, log10 Pcloud 1.5 dex, Rp 1% 
FD = dict(T=25.0, logZ=0.05, CO_ratio=0.02, log_cloudtop_P=0.10, Rp=0.002)
PRIOR_STD = dict(T=150.0, logZ=1.0, CO_ratio=0.30, log_cloudtop_P=1.50, Rp=None)


def load_systems():
    out = {}
    for r in csv.DictReader(open(PARS, encoding="utf-8")):
        f = lambda k: float(r[k]) if r[k] not in ("", None) else np.nan
        out[r["pl_name"]] = dict(
            Rs=f("st_rad") * R_sun, Mp=f("pl_bmassj") * M_jup,
            Rp=f("pl_radj") * R_jup, Teq=f("pl_eqt"),
            Rp_earth=f("pl_rade"), Mp_earth=f("pl_bmasse"),
            dens=f("pl_dens"), Teff=f("st_teff"))
    return out


def load_datasets():
    rows = list(csv.DictReader(open(SPEC, encoding="utf-8")))
    g = collections.defaultdict(list)
    for r in rows:
        ref = r["plntranreflink"] or ""
        tag = ref.split("refstr=")[-1].split(" ")[0] if "refstr=" in ref else "UNK"
        g[(r["plntname"], r["instrument"], tag)].append(r)
    ds = []
    for (pl, inst, tag), v in g.items():
        wl, dep, err = [], [], []
        for r in v:
            try:
                w = float(r["centralwavelng"]); d = float(r["plntransdep"])
                e1 = r["plntransdeperr1"]; e2 = r["plntransdeperr2"]
                if e1 in ("", None):
                    continue
                e = abs(float(e1))
                if e2 not in ("", None):
                    e = 0.5 * (abs(float(e1)) + abs(float(e2)))
            except (TypeError, ValueError):
                continue
            wl.append(w); dep.append(d / 100.0); err.append(e / 100.0)  # % -> fraction
        if len(wl) < 8:
            continue
        o = np.argsort(wl)
        ds.append(dict(planet=pl, instrument=inst, ref=tag,
                       wl_um=np.array(wl)[o], depth=np.array(dep)[o],
                       sigma=np.array(err)[o]))
    return ds


def bins_from_centres(wl_um):
    """Build contiguous bin edges from channel centres (metres)."""
    w = np.asarray(wl_um, float) * 1e-6
    mid = 0.5 * (w[1:] + w[:-1])
    lo = np.concatenate([[w[0] - (mid[0] - w[0])], mid])
    hi = np.concatenate([mid, [w[-1] + (w[-1] - mid[-1])]])
    lo = np.clip(lo, 0.31e-6, None)
    return np.array([lo, hi]).T


def main():
    systems = load_systems()
    datasets = load_datasets()
    print("datasets:", len(datasets), flush=True)

    results = []
    for k, ds in enumerate(datasets):
        pl = ds["planet"]
        if pl not in systems:
            print("skip (no system params)", pl, flush=True); continue
        sysp = systems[pl]
        uid = "%s_%s_%s_n%d" % (pl.replace(" ", ""),
                                ds["instrument"][:12].replace(" ", ""),
                                ds["ref"][:18], len(ds["wl_um"]))
        ck = os.path.join(CKDIR, "J_%s.npz" % uid)
        Tfid = float(sysp["Teq"]) if np.isfinite(sysp["Teq"]) else 1000.0
        theta0 = np.array([Tfid, 1.0, 0.53, 4.0, sysp["Rp"] / R_jup])

        if os.path.exists(ck):
            z = np.load(ck)
            J, d0 = z["J"], z["d0"]
            assert J.shape[0] == ds["sigma"].size, (ck, J.shape, ds["sigma"].shape)
            print("[%2d] cached %s" % (k, os.path.basename(ck)), flush=True)
        else:
            t0 = time.time()
            calc = TransitDepthCalculator()
            calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))

            def fwd(th):
                _, d, _ = calc.compute_depths(
                    sysp["Rs"], sysp["Mp"], th[4] * R_jup, th[0],
                    logZ=th[1], CO_ratio=th[2],
                    cloudtop_pressure=10.0 ** th[3], full_output=False)
                return np.asarray(d, float)

            d0 = fwd(theta0)
            steps = np.array([FD[n] for n in PNAMES])
            J = np.zeros((d0.size, 5))
            for i in range(5):
                tp = theta0.copy(); tp[i] += steps[i]
                tm = theta0.copy(); tm[i] -= steps[i]
                J[:, i] = (fwd(tp) - fwd(tm)) / (2 * steps[i])
            np.savez_compressed(ck, J=J, d0=d0, theta0=theta0, wl=ds["wl_um"],
                                sigma=ds["sigma"])
            print("[%2d] %-12s %-30s n=%4d  %.1fs" %
                  (k, pl, ds["instrument"][:30], d0.size, time.time() - t0), flush=True)

        # prior variances in PHYSICAL units; Rp prior = 1% of fiducial
        pstd = np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                         PRIOR_STD["log_cloudtop_P"], 0.01 * theta0[4]])
        S_a = pstd ** 2
        Kt = idf.whiten_jacobian(J, ds["sigma"], S_a)
        F = idf.fisher(Kt)
        summ = idf.resolution_matrix_summary(F, PNAMES)
        degs = idf.degeneracy_directions(F, top=2)

        results.append(dict(
            planet=pl, instrument=ds["instrument"], ref=ds["ref"],
            n_channels=int(d0.size),
            wl_um_min=float(ds["wl_um"].min()), wl_um_max=float(ds["wl_um"].max()),
            sigma_ppm_median=float(np.median(ds["sigma"]) * 1e6),
            sigma_ppm_min=float(ds["sigma"].min() * 1e6),
            depth_ppm_median=float(np.median(ds["depth"]) * 1e6),
            fiducial=dict(zip(PNAMES, [float(x) for x in theta0])),
            dofs=summ["dofs"], n_data_dominated=summ["n_data_dominated"],
            eigenvalues=summ["eigenvalues"], per_parameter=summ["per_parameter"],
            worst_direction=dict(eigenvalue=float(degs[0][0]),
                                 loading=dict(zip(PNAMES, [float(x) for x in degs[0][1]]))),
            second_worst=dict(eigenvalue=float(degs[1][0]),
                              loading=dict(zip(PNAMES, [float(x) for x in degs[1][1]]))),
        ))
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)

    print("wrote", OUT, len(results), "datasets")
    for r in sorted(results, key=lambda x: -x["dofs"]):
        print("%-11s %-26s n=%4d sig=%6.1fppm DOFS=%.2f  measured: %s"
              % (r["planet"], r["instrument"][:26], r["n_channels"],
                 r["sigma_ppm_median"], r["dofs"],
                 ",".join(p["parameter"] for p in r["per_parameter"]
                          if p["verdict"] == "measured")))


if __name__ == "__main__":
    main()
