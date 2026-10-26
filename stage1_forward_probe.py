"""
stage1_forward_probe.py
=======================
Probe the PLATON forward model: timing, parameter sensitivity, and the
Jacobian structure that the identifiability diagnostic will be built on.

NO SIMULATED DATA IS PRESENTED AS OBSERVATION ANYWHERE.
This script only characterises the forward operator F(theta) of a
publicly available radiative-transfer code (PLATON 6.3, Zhang et al. 2019/2020),
which is a deterministic model, not data.

Outputs:  experiments/stage1_probe.json
"""
import json, os, time
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_probe.json")

# ---- WASP-39 b system parameters (Faedi et al. 2011; Mancini et al. 2018) ----
# These are literature values; recorded in provenance.md with sources.
RS = 0.895 * R_sun        # stellar radius
MP = 0.28 * M_jup         # planet mass
RP = 1.279 * R_jup        # planet radius (reference)

# Fiducial atmospheric state (a plausible hot-Saturn state; NOT a retrieval result)
FID = dict(T=1100.0, logZ=1.0, CO_ratio=0.53,
           log_cloudtop_P=4.0,       # log10(Pa)
           log_scatt_factor=0.0,
           scatt_slope=4.0,
           error_multiple=1.0)

# NIRSpec PRISM-like wavelength grid: 0.6-5.3 um, R~100 binning
def prism_bins(w0=0.6e-6, w1=5.3e-6, R=100):
    edges = [w0]
    while edges[-1] < w1:
        edges.append(edges[-1] * (1 + 1.0 / R))
    edges = np.array(edges)
    return np.array([edges[:-1], edges[1:]]).T


def forward(calc, T, logZ, CO_ratio, log_cloudtop_P, Rp=RP):
    wl, depth, _ = calc.compute_depths(
        RS, MP, Rp, T, logZ=logZ, CO_ratio=CO_ratio,
        cloudtop_pressure=10.0 ** log_cloudtop_P,
        full_output=False)
    return depth


def main():
    res = {}
    t0 = time.time()
    bins = prism_bins()
    calc = TransitDepthCalculator()
    calc.change_wavelength_bins(bins)
    res["n_bins"] = int(len(bins))
    res["wl_um_min"] = float(bins[0, 0] * 1e6)
    res["wl_um_max"] = float(bins[-1, 1] * 1e6)
    res["init_seconds"] = round(time.time() - t0, 2)

    # timing of one forward evaluation on the binned grid
    ts = []
    for _ in range(5):
        t = time.time()
        d = forward(calc, FID["T"], FID["logZ"], FID["CO_ratio"], FID["log_cloudtop_P"])
        ts.append(time.time() - t)
    res["forward_seconds_median"] = float(np.median(ts))
    res["depth_ppm_mean"] = float(np.mean(d) * 1e6)
    res["depth_ppm_ptp"] = float(np.ptp(d) * 1e6)

    # ---- finite-difference Jacobian of the binned forward model ----
    # theta = [T, logZ, CO_ratio, log_cloudtop_P, Rp/R_jup]
    theta0 = np.array([FID["T"], FID["logZ"], FID["CO_ratio"],
                       FID["log_cloudtop_P"], RP / R_jup])
    steps = np.array([20.0, 0.05, 0.02, 0.10, 0.002])
    names = ["T", "logZ", "CO_ratio", "log_cloudtop_P", "Rp_Rjup"]

    d0 = forward(calc, theta0[0], theta0[1], theta0[2], theta0[3], Rp=theta0[4] * R_jup)
    J = np.zeros((len(d0), len(theta0)))
    t1 = time.time()
    for i in range(len(theta0)):
        tp = theta0.copy(); tp[i] += steps[i]
        tm = theta0.copy(); tm[i] -= steps[i]
        dp = forward(calc, tp[0], tp[1], tp[2], tp[3], Rp=tp[4] * R_jup)
        dm = forward(calc, tm[0], tm[1], tm[2], tm[3], Rp=tm[4] * R_jup)
        J[:, i] = (dp - dm) / (2 * steps[i])
    res["jacobian_seconds"] = round(time.time() - t1, 2)

    np.save(os.path.join(os.path.dirname(OUT), "stage1_J.npy"), J)
    np.save(os.path.join(os.path.dirname(OUT), "stage1_d0.npy"), d0)
    np.save(os.path.join(os.path.dirname(OUT), "stage1_bins.npy"), bins)

    # ---- diagnostic quantities at a representative JWST noise level ----
    # sigma = 50 ppm per R=100 bin is representative of NIRSpec PRISM
    # (Rustamkulov et al. 2023 report ~50-200 ppm scatter). Treated as a
    # SCENARIO here, and later replaced by the published per-bin errors.
    for sigma_ppm in (30.0, 50.0, 100.0, 200.0):
        sigma = sigma_ppm * 1e-6
        Jn = J / sigma                       # noise-normalised Jacobian
        Fisher = Jn.T @ Jn
        sv = np.linalg.svd(Jn, compute_uv=False)
        # DOFS-like measure with a unit-scaled prior: tr[(F+I)^-1 F]
        # (scaling of theta matters; we use the step sizes as the prior scale)
        S = np.diag(steps)                   # prior std proxy = FD step
        Fs = S @ Fisher @ S
        w = np.linalg.eigvalsh(Fs)
        dofs = float(np.sum(w / (1.0 + w)))
        res["sigma%d" % int(sigma_ppm)] = dict(
            singular_values=[float(x) for x in sv],
            cond=float(sv[0] / sv[-1]),
            eigen_scaled=[float(x) for x in np.sort(w)[::-1]],
            dofs=dofs,
        )

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
