import types
"""
stage6_retrieval_baseline.py
===========================
ADVERSARIAL SELF-REVIEW ITEM #1 (peer-review skill, Section A: the
"strongest baseline missing" killer). We run REAL atmospheric retrievals
with PLATON's own sampler (emcee) on REAL published JWST data, and check
whether they confirm the identifiability diagnostic's two predictions.

Predictions to test on REAL data (not synthetic injection):
  P1  IR-only spectra: metallicity posterior is prior-dominated (wide /
      centred on the prior), because DOFS<4 and prior leverage on logZ
      is high. Optical-coverage spectra should give a narrower logZ
      posterior.
  P2  Shifting the cloud-top pressure by +1 dex (across the horizon)
      translates the retrieved logZ. We test this by PINNING the cloud at
      two pressures (one below, one above the horizon ~3.0) and comparing
      the recovered logZ posterior centre.

Equilibrium chemistry (logZ + CO_ratio as the only chemistry control)
matches the diagnostic's 5-parameter state vector exactly, so the
retrieval and the diagnostic are in the same parameterisation.

INTEGRITY: every retrieved spectrum is a REAL fit to a PUBLISHED JWST
dataset. No synthetic spectrum is presented as data.
"""
import csv, json, os, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever, RetrievalResult
import emcee as _emcee
_orig_run = CombinedRetriever.run_emcee
def _safe_run_emcee(self, transit_bins, transit_depths, transit_errors,
                    eclipse_bins, eclipse_depths, eclipse_errors, fit_info,
                    nwalkers=50, nsteps=1000, include_condensation=True,
                    rad_method="xsec", num_final_samples=100, zero_opacities=[]):
    self.params_to_lnlike = {}
    initial_positions = fit_info._generate_rand_param_arrays(nwalkers)
    transit_calc = None; eclipse_calc = None
    if transit_bins is not None:
        transit_calc = TransitDepthCalculator(include_condensation=include_condensation, method=rad_method)
        transit_calc.change_wavelength_bins(transit_bins)
        self._validate_params(fit_info, transit_calc)
    sampler = _emcee.EnsembleSampler(nwalkers, fit_info._get_num_fit_params(),
        self._ln_prob, args=(transit_calc, eclipse_calc, fit_info, transit_depths,
                             transit_errors, eclipse_depths, eclipse_errors, zero_opacities))
    for i, result in enumerate(sampler.sample(initial_positions, iterations=nsteps)):
        if (i + 1) % 50 == 0:
            print("Step {}: {}".format(i + 1, self.pretty_print(fit_info)))
    best_params_arr = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    divisors, new_labels = self._get_divisors_labels(np.median(sampler.flatchain, axis=0), fit_info.fit_param_names)
    write_param_estimates_file(sampler.flatchain/divisors, best_params_arr/divisors,
                                    np.max(sampler.flatlnprobability), new_labels)
    try:
        bft, bfi, bfe, bfei = self._ln_like(best_params_arr, transit_calc, eclipse_calc,
            fit_info, transit_depths, transit_errors, eclipse_depths, eclipse_errors,
            zero_opacities=zero_opacities, ret_best_fit=True)
    except Exception:
        bft = bfi = bfe = bfei = None
    return types.SimpleNamespace(
        samples=sampler.flatchain, flatchain=sampler.flatchain,
        best_fit_params=best_params_arr, acceptance_fraction=sampler.acceptance_fraction,
        flatlnprobability=sampler.flatlnprobability,
        fit_param_names=fit_info.fit_param_names)
from platon._output_writer import write_param_estimates_file
CombinedRetriever.run_emcee = _safe_run_emcee

from collections import defaultdict

EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP, "stage6_retrieval.json")
CK = os.path.join(EXP, "ck_stage6"); os.makedirs(CK, exist_ok=True)
RS, MP, RP = 1.0, 1.0, 1.0  # in R_sun, M_jup, R_jup units (FitInfo transforms)

rows = list(csv.DictReader(open(EXP+"/transitspec_jwst.csv", encoding="utf-8")))
g = defaultdict(list)
for r in rows:
    g[(r["plntname"], r["plntranreflink"].upper())].append(r)

def load(planet_sub, ref_sub):
    key = [k for k in g if planet_sub in k[0] and ref_sub in k[1]]
    if not key: return None
    v = g[key[0]]
    wl = np.array([float(x["centralwavelng"]) for x in v])*1e-6
    d  = np.array([float(x["plntransdep"]) for x in v])/100.0
    e  = np.array([float(x["plntransdeperr1"]) for x in v])/100.0
    m  = np.isfinite(wl)&np.isfinite(d)&np.isfinite(e)&(e>0)&(wl>0.5e-6)
    wl,d,e = wl[m],d[m],e[m]
    o = np.argsort(wl)
    wl,d,e = wl[o],d[o],e[o]
    edges = np.concatenate([[wl[0]], (wl[:-1]+wl[1:])/2, [wl[-1]]])
    bins = np.column_stack([edges[:-1], edges[1:]])
    bd, be = [], []
    for i in range(len(edges)-1):
        s=(wl>=edges[i])&(wl<edges[i+1])
        if s.sum()>=1:
            w=1/e[s]**2; bd.append((d[s]*w).sum()/w.sum()); be.append(1/np.sqrt(w.sum()))
    return np.array(bins), np.array(bd), np.array(be)

def fit_eq(pin_cloud=None):
    fi = FitInfo({"Rs":1.0,"Mp":1.0,"Rp":1.0,"T":900.0,"logZ":0.0,
                  "CO_ratio":0.53,"log_cloudtop_P":4.0,"log_k":None,
                  "log_number_density":float('-inf'),"log_scatt_factor":0.0,
                  "scatt_slope":4.0,"log_part_size":-6.0,"frac_scale_height":1.0,
                  "profile_type":1.0,"n":None,"offset_transit":0.0,"offset_start":0.0,
                  "offset_end":0.0,"offset_eclipse":0.0,"error_multiple":1.0,
                  "T_star":5700.0,"T_spot":5700.0,"spot_cov_frac":0.0,
                  "log_P_quench":-99.0,"log_CH4_mult":0.0,"fit_clr":False,"fit_vmr":False})
    fi.add_uniform_fit_param("Rs",0.8,1.2,0.9,1.1)
    fi.add_uniform_fit_param("Mp",0.5,2.0,0.9,1.1)
    fi.add_uniform_fit_param("Rp",0.5,1.5,0.9,1.1)
    fi.add_uniform_fit_param("T",500,1500,700,1100)
    fi.add_uniform_fit_param("logZ",-2,3,-0.5,1.5)
    fi.add_uniform_fit_param("CO_ratio",0.2,2.0,0.3,0.8)
    if pin_cloud is None:
        fi.add_uniform_fit_param("log_cloudtop_P",2.0,6.0,2.5,5.5)
    else:
        fi.add_uniform_fit_param("log_cloudtop_P",pin_cloud-1.0,pin_cloud+1.0,pin_cloud-0.8,pin_cloud+0.8)  # pinned (±1 dex) for walker diversity
        fi.all_params["log_cloudtop_P"].best_guess = float(pin_cloud)
    return fi

def run(label, planet_sub, ref_sub, pin_cloud=None, nwalk=28, nsteps=700):
    L = load(planet_sub, ref_sub)
    if L is None: return {"label":label,"status":"no-data"}
    bins,bd,be = L
    calc = TransitDepthCalculator(); calc.change_wavelength_bins(bins)
    fi = fit_eq(pin_cloud)
    ret = CombinedRetriever()
    t0=time.time()
    try:
        res = ret.run_emcee(bins,bd,be,None,None,None,fi,
                            nwalkers=nwalk,nsteps=nsteps,
                            include_condensation=False,rad_method="xsec")
    except Exception as ex:
        return {"label":label,"status":"error","msg":str(ex)[:200]}
    samp = res.samples
    iZ = fi.fit_param_names.index("logZ")
    Z = samp[:,iZ]
    out = {"label":label,"status":"ok","elapsed_s":round(time.time()-t0,1),
           "n_samples":int(len(Z)),
           "logZ_median":float(np.median(Z)),
           "logZ_16":float(np.percentile(Z,16)),
           "logZ_84":float(np.percentile(Z,84)),
           "logZ_IQR":float(np.percentile(Z,84)-np.percentile(Z,16)),
           "logZ_prior_width":5.0,
           "pin_cloud":pin_cloud}
    np.save(os.path.join(CK, label.replace(" ","_")+".npy"), samp)
    return out

results=[]
def add(r):
    results.append(r); json.dump(results, open(OUT,"w"), indent=1)
    print(json.dumps(r, indent=1))
# P1: IR-only (Alderson G395H) equilibrium -> expect wide/prior-dominated logZ
add(run("WASP39b_IRonly_eq","WASP-39","ALDERSON"))
# P1b: optical (Feinstein NIRISS) equilibrium -> expect narrower logZ
try:
    add(run("WASP39b_optical_eq","WASP-39","FEINSTEIN"))
except Exception as ex:
    add({"label":"WASP39b_optical_eq","status":"error","msg":str(ex)[:160]})
# P2: IR-only, cloud pinned BELOW horizon (3.0) vs ABOVE (4.5)
for pc,lbl in [(3.0,"WASP39b_IR_pin3.0"),(4.5,"WASP39b_IR_pin4.5")]:
    try:
        add(run(lbl,"WASP-39","ALDERSON",pin_cloud=pc))
    except Exception as ex:
        add({"label":lbl,"status":"error","msg":str(ex)[:160]})
r_below=[x for x in results if x["label"]=="WASP39b_IR_pin3.0"]
r_above=[x for x in results if x["label"]=="WASP39b_IR_pin4.5"]
if r_below and r_above and r_below[0].get("status")=="ok" and r_above[0].get("status")=="ok":
    results.append({"label":"P2_cloud_shift_effect","status":"derived",
                    "logZ_shift_dex": round(r_above[0]["logZ_median"]-r_below[0]["logZ_median"],3),
                    "logZ_below_med":r_below[0]["logZ_median"],
                    "logZ_above_med":r_above[0]["logZ_median"]})
    json.dump(results, open(OUT,"w"), indent=1)
print("=== FINAL stage6 results ===")
print(json.dumps(results, indent=1))
