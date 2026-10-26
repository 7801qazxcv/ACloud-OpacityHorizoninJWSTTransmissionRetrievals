"""
stage3_cloud_probe.py
=====================
ADVERSARIAL SELF-CHECK, not a headline result.

stage2 found prior_leverage(log Pcloud) == 1.000 exactly, with a Fisher
eigenvalue of 0.000, for every dataset that lacks optical coverage. An exact
zero is suspicious: it may be an artefact of evaluating the Jacobian at a
fiducial cloud-top pressure (1e4 Pa) that happens to sit below the 3-5 um
photosphere, rather than a statement about the measurement.

This script separates the two explanations by sweeping the fiducial cloud-top
pressure over 1e-2 .. 1e7 Pa for one optical-inclusive dataset
(Rustamkulov PRISM, 0.53-5.34 um) and one IR-only dataset
(Alderson G395H, 2.76-5.17 um), using each dataset's real sigma.

If the IR-only curve is zero at ALL cloud pressures, the degeneracy is a
property of the wavelength coverage (real result).
If it becomes nonzero at some pressures, stage2's claim must be restated as
conditional on the fiducial state (weaker result, honestly reported).

Outputs: stage3_cloud_probe.json
"""
import csv, collections, json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup
import identifiability as idf
from stage2_real_diagnostics import (load_systems, load_datasets, bins_from_centres,
                                     PNAMES, FD, PRIOR_STD)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage3_cloud_probe.json")

TARGETS = [("WASP-39 b", "RUSTAMKULOV", "optical+IR"),
           ("WASP-39 b", "ALDERSON", "IR-only"),
           ("GJ 486 b", "MORAN", "IR-only")]
LOGPC = np.arange(-2.0, 7.01, 0.5)


def main():
    systems, datasets = load_systems(), load_datasets()
    res = []
    for pl, reftag, label in TARGETS:
        cand = [d for d in datasets if d["planet"] == pl and reftag in d["ref"]]
        if not cand:
            print("no dataset for", pl, reftag); continue
        ds = max(cand, key=lambda d: len(d["wl_um"]))
        sysp = systems[pl]
        calc = TransitDepthCalculator()
        calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))
        Tfid = float(sysp["Teq"])

        def fwd(th):
            _, d, _ = calc.compute_depths(
                sysp["Rs"], sysp["Mp"], th[4] * R_jup, th[0],
                logZ=th[1], CO_ratio=th[2],
                cloudtop_pressure=10.0 ** th[3], full_output=False)
            return np.asarray(d, float)

        steps = np.array([FD[n] for n in PNAMES])
        pstd = np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                         PRIOR_STD["log_cloudtop_P"], 0.01 * sysp["Rp"] / R_jup])
        S_a = pstd ** 2
        curve = []
        t0 = time.time()
        for lpc in LOGPC:
            th0 = np.array([Tfid, 1.0, 0.53, float(lpc), sysp["Rp"] / R_jup])
            J = np.zeros((len(ds["wl_um"]), 5))
            for i in range(5):
                tp = th0.copy(); tp[i] += steps[i]
                tm = th0.copy(); tm[i] -= steps[i]
                J[:, i] = (fwd(tp) - fwd(tm)) / (2 * steps[i])
            F = idf.fisher(idf.whiten_jacobian(J, ds["sigma"], S_a))
            A = idf.averaging_kernel(F)
            lam = np.sort(np.linalg.eigvalsh(F))[::-1]
            curve.append(dict(log_cloudtop_P=float(lpc),
                              dofs=float(idf.dofs(F)),
                              A_diag_cloud=float(A[3, 3]),
                              leverage_cloud=float(1 - A[3, 3]),
                              lam_min=float(lam[-1]),
                              fisher_cloud_diag=float(F[3, 3])))
            print("%-11s %-11s logPc=%+.1f dofs=%.3f L_cloud=%.4f F_cc=%.3e"
                  % (pl, label, lpc, curve[-1]["dofs"], curve[-1]["leverage_cloud"],
                     curve[-1]["fisher_cloud_diag"]), flush=True)
        res.append(dict(planet=pl, ref=ds["ref"], label=label,
                        wl_um=[float(ds["wl_um"].min()), float(ds["wl_um"].max())],
                        n_channels=len(ds["wl_um"]),
                        sigma_ppm_median=float(np.median(ds["sigma"]) * 1e6),
                        seconds=round(time.time() - t0, 1), curve=curve))
        with open(OUT, "w") as f:
            json.dump(res, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
