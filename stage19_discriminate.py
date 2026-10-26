"""
stage19_discriminate.py  --  is the cloud-offset experiment informative?
========================================================================
Integrity principle 17b: before reporting that the retrieved metallicity does
NOT shift when the cloud is moved by 1.5 dex, we must check that the two setups
are actually distinguishable.  If the two models produce spectra that differ by
much less than the measurement errors, the "null" carries no information.

We take the posterior-median parameters of two runs, build the corresponding
model spectra on the real wavelength grid of the spectrum they were fitted to,
and report:
  * the rms and maximum difference between the two model spectra in units of
    the published per-channel sigma,
  * the difference in their ln-likelihood (chi^2/2) on that same data,
  * for the frozen-cloud pair, the same quantity for the cloud alone, i.e. how
    much the predicted spectrum changes when only the cloud top moves.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_jup
import stage10_retrieval_long as S


def param_dict(label):
    meta = json.load(open(os.path.join(HERE, "stage10_%s.json" % label),
                          encoding="utf-8"))
    post = np.load(os.path.join(HERE, "ck_stage10", label + "_chain.npy"))
    names = list(meta["stats"].keys())
    flat = post[meta["burn_in_steps"]:].reshape(-1, post.shape[2])
    med = np.median(flat, axis=0)
    d = dict(names=list(meta["stats"].keys()), median=med,
             cloud=meta.get("cloud_frozen_at"), CO=meta.get("CO_frozen_at"))
    return d


def model_at(label, d, override=None, ref="ALDERSON"):
    L = S.load("WASP-39", ref)
    bins, bd, be = L
    calc = TransitDepthCalculator(include_condensation=False, method="xsec")
    calc.change_wavelength_bins(bins)
    sy = S.SYSTEMS["WASP-39"]
    kw = dict(logZ=0.0, CO_ratio=0.53, cloudtop_pressure=1e4)
    vals = dict(zip(d["names"], d["median"]))
    Rp = vals["Rp"]
    T = vals.get("T", 900.0)
    logZ = vals.get("logZ", 0.0)
    CO = vals.get("CO_ratio", 0.53)
    pc = 10.0 ** vals.get("log_cloudtop_P", 4.0)
    if d["CO"] is not None:
        CO = d["CO"]
    if d["cloud"] is not None:
        pc = 10.0 ** d["cloud"]
    if override:
        T = override.get("T", T); logZ = override.get("logZ", logZ)
        CO = override.get("CO", CO); pc = override.get("pc", pc)
        Rp = override.get("Rp", Rp)
    _, dd, _ = calc.compute_depths(sy["Rs"], sy["Mp"], Rp, T, logZ=logZ,
                                   CO_ratio=CO, cloudtop_pressure=pc)
    return np.asarray(dd, float), bd, be


def chi2(model, bd, be):
    return float((((model - bd) / be) ** 2).sum())


def main():
    pairs = [("IRonly_free", "optical_free"),
             ("IR_cloudfix3.0", "IR_cloudfix4.5"),
             ("IR_CO053_cloud3.0", "IR_CO053_cloud4.5")]
    out = {}
    for a, b in pairs:
        if not os.path.exists(os.path.join(HERE, "ck_stage10", a + "_chain.npy")) \
           or not os.path.exists(os.path.join(HERE, "ck_stage10", b + "_chain.npy")):
            print("skip (missing chains):", a, b)
            continue
        da, db = param_dict(a), param_dict(b)
        ref = "FEINSTEIN" if "optical" in b else "ALDERSON"
        ma, bd, be = model_at(a, da, ref=ref)
        mb, _, _ = model_at(b, db, ref=ref)
        diff = (mb - ma) / be
        rec = dict(pair=[a, b],
                   rms_diff_in_sigma=float(np.sqrt(np.mean(diff ** 2))),
                   max_diff_in_sigma=float(np.max(np.abs(diff))),
                   chi2_a=chi2(ma, bd, be), chi2_b=chi2(mb, bd, be),
                   n_data=int(len(bd)))
        print("%-20s vs %-20s  rms diff = %.2f sigma, max = %.2f sigma, "
              "chi2 %8.1f vs %8.1f (n=%d)"
              % (a, b, rec["rms_diff_in_sigma"], rec["max_diff_in_sigma"],
                 rec["chi2_a"], rec["chi2_b"], rec["n_data"]))
        out[a + "__" + b] = rec
    json.dump(out, open(os.path.join(HERE, "stage19_discriminate.json"), "w",
                        encoding="utf-8"), indent=1)
    print("wrote stage19_discriminate.json")


if __name__ == "__main__":
    main()
