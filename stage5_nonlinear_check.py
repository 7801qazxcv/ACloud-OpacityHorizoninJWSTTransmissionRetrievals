"""
stage5_nonlinear_check.py
=========================
Validate the LINEAR bias-transfer operator used in stage4 against a fully
NONLINEAR retrieval-style refit. Reviewer's obvious objection: a 1-dex error in
log Pc is not infinitesimal, so a Jacobian-based bias estimate may be wrong.

Procedure (per dataset, at log Pc = 2.5, i.e. inside the horizon):
  1. Generate a noiseless MODEL spectrum at theta_true (cloud at 10^2.5 Pa).
     -- this is a self-consistency test of the estimator, NOT an observation.
  2. Fix log Pc at theta_true + delta (delta = 0.25, 0.5, 1.0 dex) and refit the
     remaining 4 parameters by damped Gauss-Newton to convergence.
  3. Compare the recovered logZ offset with the linear prediction from stage4.

This is an injection-recovery test of the DIAGNOSTIC, run on model spectra.
Nowhere is a model spectrum presented as data. The real spectra enter only
through their published per-channel sigma, which sets the weighting.

Outputs: stage5_nonlinear.json
"""
import json, os, sys, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from platon.constants import R_jup
import identifiability as idf
from stage2_real_diagnostics import (load_systems, load_datasets, PNAMES, FD, PRIOR_STD)
from stage4_horizon_and_bias import make_fwd, jac, prior_cov

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage5_nonlinear.json")
DELTAS = [0.25, 0.5, 1.0]
LPC_TRUE = 2.5
# one representative dataset per (planet, coverage) class, to keep runtime sane
PICK = [("WASP-39 b", "ALDERSON"), ("WASP-39 b", "RUSTAMKULOV"),
        ("WASP-39 b", "FEINSTEIN"), ("GJ 486 b", "MORAN"), ("HAT-P-18 b", "FU_ET_AL")]


def refit_fixed_cloud(fwd, ds, sysp, th_true, d_obs, lpc_fixed, maxit=40):
    """Damped Gauss-Newton MAP fit of [T, logZ, CO, Rp] with log Pc held fixed."""
    S_a = prior_cov(sysp)
    free = [0, 1, 2, 4]
    th = th_true.copy()
    th[3] = lpc_fixed
    sig = ds["sigma"]
    steps = np.array([FD[n] for n in PNAMES])
    lam = 1.0
    prev = np.inf
    for it in range(maxit):
        r = d_obs - fwd(th)
        chi2 = float(np.sum((r / sig) ** 2))
        # Jacobian on free parameters only
        J = np.zeros((d_obs.size, len(free)))
        for a, i in enumerate(free):
            tp = th.copy(); tp[i] += steps[i]
            tm = th.copy(); tm[i] -= steps[i]
            J[:, a] = (fwd(tp) - fwd(tm)) / (2 * steps[i])
        Kn = J / sig[:, None]
        rn = r / sig
        Sa_inv = np.diag(1.0 / S_a[free])
        dth_a = th[free] - np.array([th_true[0], 1.0, 0.53, th_true[4]])
        H = Kn.T @ Kn + lam * Sa_inv
        g = Kn.T @ rn - Sa_inv @ dth_a
        step = np.linalg.solve(H, g)
        # cap the step so the forward model stays in its validity range
        cap = np.array([200.0, 0.5, 0.15, 0.05])
        step = np.clip(step, -cap, cap)
        th_new = th.copy()
        th_new[free] = th[free] + step
        th_new[0] = np.clip(th_new[0], 300.0, 3000.0)
        th_new[1] = np.clip(th_new[1], -1.0, 3.0)
        th_new[2] = np.clip(th_new[2], 0.05, 2.0)
        chi2_new = float(np.sum(((d_obs - fwd(th_new)) / sig) ** 2))
        if chi2_new < chi2:
            th, lam = th_new, max(lam * 0.5, 1e-3)
        else:
            lam = min(lam * 4.0, 1e4)
        if abs(prev - chi2_new) < 1e-6 * max(chi2_new, 1.0):
            break
        prev = chi2_new
    return th, chi2_new, it + 1


def main():
    systems, datasets = load_systems(), load_datasets()
    res = []
    for pl, tag in PICK:
        cand = [d for d in datasets if d["planet"] == pl and tag in d["ref"]]
        if not cand:
            print("missing", pl, tag, flush=True); continue
        ds = max(cand, key=lambda d: len(d["wl_um"]))
        sysp = systems[pl]
        fwd = make_fwd(sysp, ds)
        th_true = np.array([float(sysp["Teq"]), 1.0, 0.53, LPC_TRUE, sysp["Rp"] / R_jup])
        d_true = fwd(th_true)
        # linear prediction from the same Jacobian machinery
        J = jac(fwd, th_true)
        free, blin = idf.bias_from_fixing(J, ds["sigma"], prior_cov(sysp),
                                          idx_fixed=[3], delta=[1.0])
        lin = {PNAMES[j]: float(b) for j, b in zip(free, blin)}
        rows = []
        for dl in DELTAS:
            t0 = time.time()
            th_hat, chi2, nit = refit_fixed_cloud(fwd, ds, sysp, th_true, d_true,
                                                  LPC_TRUE + dl)
            rows.append(dict(delta_dex=dl, iters=nit, chi2=chi2,
                             nonlinear=dict(T=float(th_hat[0] - th_true[0]),
                                            logZ=float(th_hat[1] - th_true[1]),
                                            CO_ratio=float(th_hat[2] - th_true[2]),
                                            Rp=float(th_hat[4] - th_true[4])),
                             linear_scaled=dict((k, v * dl) for k, v in lin.items()),
                             seconds=round(time.time() - t0, 1)))
            print("%-11s %-12s d=%.2f  nonlin dlogZ=%+.3f  lin dlogZ=%+.3f  chi2=%.3g  %.0fs"
                  % (pl, tag[:12], dl, rows[-1]["nonlinear"]["logZ"],
                     rows[-1]["linear_scaled"]["logZ"], chi2, rows[-1]["seconds"]), flush=True)
        res.append(dict(planet=pl, ref=ds["ref"], n_channels=len(ds["wl_um"]),
                        has_optical=bool(ds["wl_um"].min() < 1.0),
                        sigma_ppm_median=float(np.median(ds["sigma"]) * 1e6),
                        linear_per_dex=lin, tests=rows))
        json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
