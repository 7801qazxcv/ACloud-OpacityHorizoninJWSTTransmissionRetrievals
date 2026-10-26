"""
stage10_retrieval_long.py  --  REVISION ROUND 1 (referee items #17,#18,#20)
===========================================================================
Independent equilibrium-chemistry emcee retrievals on PUBLISHED JWST spectra,
run at full cost with PLATON's own ensemble sampler.

History / why this replaces the round-1 script
----------------------------------------------
v1 of this pipeline (stage6_retrieval.py) was silently invalid: it handed
PLATON's FitInfo the values Rs=1.0, Mp=1.0, Rp=1.0 with uniform priors
0.8<Rs<1.2 etc.  PLATON's FitInfo/CombinedRetriever work in SI units (metres,
kg -- see platon/combined_retriever.py, which divides by R_sun, M_jup, R_jup
only when *printing*).  With a 1-metre star every forward model raised
`AtmosphereError: Atmosphere unbound`, so every walker had ln p = -inf, emcee
never accepted a move, and the saved "chain" was the walkers' initial draws
repeated.  Two unseeded re-runs of that script returned different medians
(0.74 / 0.40 / 0.54 dex) -- the signature of a chain that never moved.

v2 fixed the units but used prior ranges that truncated the posteriors: with
2 < log P_c < 6 and 0.2 < C/O < 2.0 the WASP-39b G395H posterior piled up
against BOTH lower edges (logP_c = 2.04 +/- 0.04, C/O = 0.218 +/- 0.026), so
the reported intervals were prior artefacts.  v3 (this file) therefore uses
deliberately wide uniform priors, and tests the cloud offsets with the cloud
parameter *frozen* (removed from the fitted set) rather than softly pinned.

Design of the four runs
-----------------------
  IRonly_free    IR-only spectrum, all five parameters free, wide priors
  optical_free   optical+NIR spectrum, identical setup (the coverage test)
  IR_cloudfix3.0 cloud frozen at 10^3.0 Pa  (below the IR-only horizon 3.37)
  IR_cloudfix4.5 cloud frozen at 10^4.5 Pa  (above the horizon)
The last two differ by exactly 1.5 dex in the cloud, so the difference in their
metallicity posteriors is a direct measurement of the bias transfer predicted
by Equation (7).

INTEGRITY: every run fits PUBLISHED transit depths.  Nothing synthetic.
"""
import csv, json, os, sys, time, warnings
from collections import defaultdict
import numpy as np
warnings.filterwarnings("ignore")

from platon.transit_depth_calculator import TransitDepthCalculator
from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever
from platon.constants import R_sun, M_jup, R_jup
import emcee as _emcee

EXP = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(EXP, "ck_stage10"); os.makedirs(CK, exist_ok=True)

NSTEPS = int(os.environ.get("PLATON_NSTEPS", 600))
NWALK = int(os.environ.get("PLATON_NWALK", 24))
BURN_FRAC = 0.5

# Literature system parameters in SI (NASA Exoplanet Archive; paper Table 3)
SYSTEMS = {
    "WASP-39": dict(Rs=0.939 * R_sun, Mp=0.281 * M_jup, Rp0=1.279 * R_jup,
                    Tstar=5485.0),
}

PNAMES = ["Rp", "T", "logZ", "CO_ratio", "log_cloudtop_P"]
PNAMES_NOCLOUD = ["Rp", "T", "logZ", "CO_ratio"]
# prior ranges actually used for the uniform priors below
PRIOR_RANGE = {"Rp": None,            # filled per planet (+/-10%)
               "T": (400.0, 1600.0),
               "logZ": (-2.0, 3.0),
               "CO_ratio": (0.05, 2.0),
               "log_cloudtop_P": (0.0, 6.0)}

rows_csv = list(csv.DictReader(open(EXP + "/transitspec_jwst.csv", encoding="utf-8")))
g = defaultdict(list)
for _r in rows_csv:
    g[(_r["plntname"], _r["plntranreflink"].upper())].append(_r)


def load(planet_sub, ref_sub):
    """Published spectrum -> (bins[metres], depth fraction, error fraction)."""
    key = [k for k in g if planet_sub in k[0] and ref_sub in k[1]]
    if not key:
        return None
    v = g[key[0]]
    wl = np.array([float(x["centralwavelng"]) for x in v]) * 1e-6
    d = np.array([float(x["plntransdep"]) for x in v]) / 100.0
    e = np.array([abs(float(x["plntransdeperr1"])) for x in v]) / 100.0
    m = np.isfinite(wl) & np.isfinite(d) & np.isfinite(e) & (e > 0) & (wl > 0.5e-6)
    wl, d, e = wl[m], d[m], e[m]
    o = np.argsort(wl)
    wl, d, e = wl[o], d[o], e[o]
    edges = np.concatenate([[wl[0]], (wl[:-1] + wl[1:]) / 2, [wl[-1]]])
    edges[-1] = edges[-1] * (1 + 1e-9)
    bins, bd, be = [], [], []
    for i in range(len(edges) - 1):
        s = (wl >= edges[i]) & (wl < edges[i + 1])
        if s.sum() >= 1:
            w = 1 / e[s] ** 2
            bins.append([edges[i], edges[i + 1]])
            bd.append((d[s] * w).sum() / w.sum()); be.append(1 / np.sqrt(w.sum()))
    return np.array(bins), np.array(bd), np.array(be)


def fit_info(planet, fix_cloud=None, fix_CO=None):
    """Five free parameters with wide uniform priors; if fix_cloud / fix_CO are
    numbers those parameters are removed from the fitted set and frozen."""
    sy = SYSTEMS[planet]
    p = dict(Rs=sy["Rs"], Mp=sy["Mp"], Rp=sy["Rp0"], T=900.0, logZ=0.0,
             CO_ratio=0.53, log_cloudtop_P=4.0,
             log_k=None, log_number_density=float('-inf'), log_scatt_factor=0.0,
             scatt_slope=4.0, log_part_size=-6.0, frac_scale_height=1.0,
             T_star=sy["Tstar"], T_spot=sy["Tstar"], spot_cov_frac=0.0,
             log_P_quench=-99.0, log_CH4_mult=0.0, fit_clr=False, fit_vmr=False,
             profile_type=1.0, n=None, offset_transit=0, offset_start=0,
             offset_end=0, offset_eclipse=0, error_multiple=1.0)
    fi = FitInfo(p)
    Rp0 = sy["Rp0"]
    fi.add_uniform_fit_param("Rp", Rp0 * 0.9, Rp0 * 1.1, Rp0 * 0.95, Rp0 * 1.05)
    fi.add_uniform_fit_param("T", *PRIOR_RANGE["T"], 700, 1100)
    fi.add_uniform_fit_param("logZ", *PRIOR_RANGE["logZ"], -0.5, 1.5)
    if fix_CO is None:
        fi.add_uniform_fit_param("CO_ratio", *PRIOR_RANGE["CO_ratio"], 0.3, 0.8)
    else:
        fi.all_params["CO_ratio"].best_guess = float(fix_CO)
    if fix_cloud is None:
        fi.add_uniform_fit_param("log_cloudtop_P", *PRIOR_RANGE["log_cloudtop_P"],
                                 2.5, 5.5)
    else:
        fi.all_params["log_cloudtop_P"].best_guess = float(fix_cloud)
    return fi


def names_for(fix_cloud, fix_CO=None):
    nm = ["Rp", "T", "logZ", "CO_ratio", "log_cloudtop_P"]
    if fix_CO is not None:
        nm.remove("CO_ratio")
    if fix_cloud is not None:
        nm.remove("log_cloudtop_P")
    return nm


def neg_lnprob(x, ret, calc, fi, bd, be):
    try:
        v = ret._ln_prob(x, calc, None, fi, bd, be, None, None, [])
    except Exception:
        return np.inf
    return -v if np.isfinite(v) else np.inf


def make_sampler(fi, bins, bd, be, nwalkers):
    ret = CombinedRetriever()
    ret.params_to_lnlike = {}
    calc = TransitDepthCalculator(include_condensation=False, method="xsec")
    calc.change_wavelength_bins(bins)
    ret._validate_params(fi, calc)
    sampler = _emcee.EnsembleSampler(
        nwalkers, fi._get_num_fit_params(), ret._ln_prob,
        args=(calc, None, fi, bd, be, None, None, []))
    return ret, calc, sampler


def start_positions(fi, bins, bd, be, nwalkers):
    """Nelder-Mead MAP followed by a ball of walkers around it; every start
    must have a finite log-probability (the failure mode that broke v1)."""
    from scipy.optimize import minimize
    ret, calc, sampler = make_sampler(fi, bins, bd, be, nwalkers)
    nm = fi.fit_param_names
    lo = np.array([fi.all_params[n].low_lim for n in nm])
    hi = np.array([fi.all_params[n].high_lim for n in nm])
    x0 = np.clip(np.array([fi.all_params[n].best_guess for n in nm]),
                 lo + 1e-6, hi - 1e-6)
    res = minimize(neg_lnprob, x0, args=(ret, calc, fi, bd, be),
                   method="Nelder-Mead",
                   options=dict(maxiter=600, xatol=1e-5, fatol=1e-3))
    x_map = np.clip(res.x, lo + 1e-6, hi - 1e-6)
    lp_map = float(-res.fun)
    rel = np.full(len(nm), 0.02)          # 2% of the prior range
    rel[np.array([nm.index("logZ") if "logZ" in nm else 0])] = 0.01
    scale = rel * (hi - lo)
    rng = np.random.default_rng(20261020)
    pos, lp = None, None
    for attempt in range(12):
        cand = np.clip(x_map[None, :] + rng.normal(size=(nwalkers, len(nm)))
                       * scale[None, :] * (0.5 ** attempt), lo[None, :] + 1e-9,
                       hi[None, :] - 1e-9)
        cand[0] = np.clip(x_map, lo + 1e-9, hi - 1e-9)
        lpc = np.array([ret._ln_prob(c, calc, None, fi, bd, be, None, None, [])
                        for c in cand])
        if np.all(np.isfinite(lpc)):
            pos, lp = cand, lpc
            break
    if pos is None:
        raise RuntimeError("could not start all walkers at finite lnprob")
    return ret, calc, sampler, pos, x_map, lp_map


def run(label, planet_sub, ref_sub, fix_cloud=None, fix_CO=None,
        nwalkers=NWALK, nsteps=NSTEPS):
    L = load(planet_sub, ref_sub)
    bins, bd, be = L
    fi = fit_info(planet_sub, fix_cloud, fix_CO)
    ret, calc, sampler, pos, x_map, lp_map = start_positions(fi, bins, bd, be, nwalkers)
    print("  MAP: lnP=%.2f  x=%s" % (lp_map, np.array2string(x_map, precision=3)),
          flush=True)
    t0 = time.time()
    for i, _ in enumerate(sampler.sample(pos, iterations=nsteps)):
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print("  %-18s step %4d/%d  %.0fs  eta %.0fs"
                  % (label, i + 1, nsteps, el, el / (i + 1) * (nsteps - i - 1)),
                  flush=True)
    chain = sampler.get_chain()
    lnpr = sampler.get_log_prob()
    acc = float(np.mean(sampler.acceptance_fraction))
    return dict(label=label, chain=chain, lnpr=lnpr, acc=acc,
                elapsed=time.time() - t0, fi=fi, x_map=x_map, lp_map=lp_map,
                bins=bins, bd=bd, be=be)


def summarize(flat, names, planet):
    Rp0 = SYSTEMS[planet]["Rp0"]
    bounds = {"Rp": (0.9 * Rp0 / R_jup, 1.1 * Rp0 / R_jup),
              "T": PRIOR_RANGE["T"], "logZ": PRIOR_RANGE["logZ"],
              "CO_ratio": PRIOR_RANGE["CO_ratio"],
              "log_cloudtop_P": PRIOR_RANGE["log_cloudtop_P"]}
    out = {}
    for j, nm in enumerate(names):
        x = flat[:, j] / R_jup if nm == "Rp" else flat[:, j]
        lo, hi = bounds[nm]
        q = np.percentile(x, [16, 50, 84])
        p10, p90 = np.percentile(x, [10, 90])
        # how much of the prior range the posterior occupies
        out[nm] = dict(median=float(q[1]), lo68=float(q[0]), hi68=float(q[2]),
                       iqr=float(q[2] - q[0]), prior_lo=float(lo),
                       prior_hi=float(hi), prior_width=float(hi - lo),
                       iqr_frac=float((q[2] - q[0]) / (hi - lo)),
                       p10_p90_frac=float((p90 - p10) / (hi - lo)),
                       dist_to_lower_edge=float((q[1] - lo) / (hi - lo)),
                       mean=float(x.mean()), std=float(x.std(ddof=1)))
    return out


def ess(chain, names):
    try:
        tau = _emcee.autocorr.integrated_time(chain, quiet=True)
    except Exception as ex:
        return {"error": str(ex)[:120]}
    return {nm: dict(tau_steps=round(float(tau[j]), 1),
                     ess=int(chain.shape[0] * chain.shape[1] / float(tau[j])))
            for j, nm in enumerate(names)}


CONFIGS = [
    ("IRonly_free", "WASP-39", "ALDERSON", None, None),
    ("optical_free", "WASP-39", "FEINSTEIN", None, None),
    ("IR_cloudfix3.0", "WASP-39", "ALDERSON", 3.0, None),
    ("IR_cloudfix4.5", "WASP-39", "ALDERSON", 4.5, None),
    # diagnostics-controlled pair: C/O frozen at solar so that the two runs
    # differ ONLY in the assumed cloud-top pressure, with no parameter sitting
    # against a prior edge (this is the test of the bias-transfer operator)
    ("IR_CO053_cloud3.0", "WASP-39", "ALDERSON", 3.0, 0.53),
    ("IR_CO053_cloud4.5", "WASP-39", "ALDERSON", 4.5, 0.53),
    # pressure pair where the model actually responds to the cloud (see
    # stage19b): only here can the bias transfer be measured rather than not
    ("IR_CO053_cloud2.0", "WASP-39", "ALDERSON", 2.0, 0.53),
    ("IR_CO053_cloud2.5", "WASP-39", "ALDERSON", 2.5, 0.53),
]


def main():
    labels = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    for label, planet, ref, fix, fixco in CONFIGS:
        if labels and label not in labels:
            continue
        out = os.path.join(EXP, "stage10_%s.json" % label)
        chainfile = os.path.join(CK, label + "_chain.npy")
        print("RUN", label, flush=True)
        try:
            r = run(label, planet, ref, fix_cloud=fix, fix_CO=fixco)
        except Exception as ex:
            print("  ERROR", str(ex)[:300], flush=True)
            json.dump({"label": label, "status": "error", "msg": str(ex)[:300]},
                      open(out, "w", encoding="utf-8"), indent=1)
            continue
        np.save(chainfile, r["chain"])
        np.save(chainfile.replace("_chain.npy", "_lnprob.npy"), r["lnpr"])
        names = names_for(fix, fixco)
        nb = int(r["chain"].shape[0] * BURN_FRAC)
        post = r["chain"][nb:].reshape(-1, r["chain"].shape[2])
        rec = {"label": label, "status": "ok", "planet": planet, "ref": ref,
               "cloud_frozen_at": fix, "CO_frozen_at": fixco,
               "n_channels": int(len(r["bd"])),
               "nwalkers": int(r["chain"].shape[1]),
               "nsteps": int(r["chain"].shape[0]),
               "burn_in_steps": nb, "thin": 1,
               "acceptance_fraction": r["acc"],
               "elapsed_s": round(r["elapsed"], 1),
               "map_lnprob": r["lp_map"],
               "map_point": dict(zip(names, [float(x) for x in r["x_map"]])),
               "n_samples_after_burnin": int(post.shape[0]),
               "ess": ess(r["chain"][nb:], names),
               "stats": summarize(post, names, planet)}
        # correlations (only for parameters present)
        idx = {nm: names.index(nm) for nm in names}
        cors = {}
        for a in names:
            for b in names:
                if a < b:
                    cors["rho_%s_%s" % (a, b)] = float(
                        np.corrcoef(post[:, idx[a]], post[:, idx[b]])[0, 1])
        rec["correlations"] = cors
        json.dump(rec, open(out, "w", encoding="utf-8"), indent=1)
        print("  DONE %s acc=%.2f  ESS=%s" %
              (label, r["acc"], {k: v["ess"] for k, v in rec["ess"].items()}), flush=True)
        for nm in names:
            s = rec["stats"][nm]
            print("    %-15s %.4f [%.4f, %.4f]  prior_frac=%.3f  frac_to_low_edge=%.3f"
                  % (nm, s["median"], s["lo68"], s["hi68"], s["iqr_frac"],
                     s["dist_to_lower_edge"]), flush=True)


if __name__ == "__main__":
    main()
