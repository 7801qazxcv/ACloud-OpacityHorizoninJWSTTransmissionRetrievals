"""
stage19b_cloud_sensitivity.py  --  does the cloud matter AT the retrieved state?
===============================================================================
stage19 showed that two models differing only by 1.5 dex in cloud-top pressure
differ by 0.08 sigma rms, i.e. the frozen-cloud experiment has no power *at the
state the retrieval actually lands in*.  Here we locate where the cloud starts
to matter at that state: we take a posterior-median parameter vector and
compare the model with the cloud removed against the model with the cloud at a
range of pressures, in units of the published per-channel sigma.

The point is to test whether the retrieval has simply moved to a metallicity at
which Section 6.5 says the horizon has dropped below both values we tried --
which would mean the diagnostic and the retrieval agree.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from platon.transit_depth_calculator import TransitDepthCalculator
import stage10_retrieval_long as S

LOGPC_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]


def posterior_median(label):
    meta = json.load(open(os.path.join(HERE, "stage10_%s.json" % label),
                          encoding="utf-8"))
    post = np.load(os.path.join(HERE, "ck_stage10", label + "_chain.npy"))
    flat = post[meta["burn_in_steps"]:].reshape(-1, post.shape[2])
    return meta, dict(zip(meta["stats"].keys(), np.median(flat, axis=0)))


def spectra(label, cloud_pc=None, logZ=None, ref="ALDERSON"):
    meta, vals = posterior_median(label)
    L = S.load("WASP-39", ref)
    bins, bd, be = L
    calc = TransitDepthCalculator(include_condensation=False, method="xsec")
    calc.change_wavelength_bins(bins)
    sy = S.SYSTEMS["WASP-39"]
    T = vals.get("T", 900.0)
    lz = vals.get("logZ", 0.0) if logZ is None else logZ
    CO = meta.get("CO_frozen_at") or vals.get("CO_ratio", 0.53)
    Rp = vals["Rp"]
    pc = np.inf if cloud_pc is None else cloud_pc
    _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], Rp, T, logZ=lz,
                                  CO_ratio=CO, cloudtop_pressure=pc)
    return np.asarray(d, float), bd, be, dict(T=T, logZ=lz, CO=CO, Rp=Rp)


def main():
    out = {}
    for label in ("IRonly_free", "IR_CO053_cloud3.0", "optical_free"):
        if not os.path.exists(os.path.join(HERE, "ck_stage10", label + "_chain.npy")):
            print("skip", label)
            continue
        ref = "ALDERSON"
        base, bd, be, st = spectra(label, cloud_pc=np.inf, ref=ref)
        rows = []
        for lp in LOGPC_GRID:
            m, _, _, _ = spectra(label, cloud_pc=10.0 ** lp, ref=ref)
            diff = (m - base) / be
            rows.append(dict(logPc=lp,
                             rms_in_sigma=float(np.sqrt(np.mean(diff ** 2))),
                             max_in_sigma=float(np.max(np.abs(diff)))))
            print("%-18s logPc=%.1f  rms=%7.2f sigma  max=%7.2f sigma  "
                  "(state: T=%.0f K, logZ=%.2f, C/O=%.2f)"
                  % (label, lp, rows[-1]["rms_in_sigma"], rows[-1]["max_in_sigma"],
                     st["T"], st["logZ"], st["CO"]))
        out[label] = dict(state=st, rows=rows)
    json.dump(out, open(os.path.join(HERE, "stage19b_cloud_sensitivity.json"), "w",
                        encoding="utf-8"), indent=1)
    print("wrote stage19b_cloud_sensitivity.json")


if __name__ == "__main__":
    main()
