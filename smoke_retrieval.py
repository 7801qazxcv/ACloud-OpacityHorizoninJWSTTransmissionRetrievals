"""Minimal PLATON emcee retrieval smoke test (corrected field names + units).
Confirms API works and measures memory/time before full stage6."""
import csv, os, time, warnings, json
import numpy as np
warnings.filterwarnings("ignore")
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.fit_info import FitInfo
from platon.combined_retriever import CombinedRetriever, RetrievalResult
import types
_orig_run = CombinedRetriever.run_emcee
def _safe_run_emcee(self, transit_bins, transit_depths, transit_errors,
                    eclipse_bins, eclipse_depths, eclipse_errors, fit_info,
                    nwalkers=50, nsteps=1000, include_condensation=True,
                    rad_method="xsec", num_final_samples=100, zero_opacities=[]):
    """Wrapper: identical to run_emcee but tolerates a -inf MAP point
    (forward-model edge) so the posterior samples are still returned."""
    import emcee
    self.params_to_lnlike = {}
    initial_positions = fit_info._generate_rand_param_arrays(nwalkers)
    transit_calc = None; eclipse_calc = None
    if transit_bins is not None:
        transit_calc = TransitDepthCalculator(include_condensation=include_condensation, method=rad_method)
        transit_calc.change_wavelength_bins(transit_bins)
        self._validate_params(fit_info, transit_calc)
    sampler = emcee.EnsembleSampler(nwalkers, fit_info._get_num_fit_params(),
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
rows = list(csv.DictReader(open(EXP+"/transitspec_jwst.csv", encoding="utf-8")))
g = defaultdict(list)
for r in rows:
    g[(r["plntname"], r["instrument"], r["plntranreflink"][:18])].append(r)

key=None
for k in g:
    if "WASP-39" in k[0] and "ALDERSON" in k[2].upper():
        key=k
print("selected:", key, "npts:", len(g[key]))

wl=np.array([float(x["centralwavelng"]) for x in g[key]])*1e-6   # microns -> metres
d =np.array([float(x["plntransdep"]) for x in g[key]])/100.0   # % -> fraction
e =np.array([float(x["plntransdeperr1"]) for x in g[key]])/100.0
m=np.isfinite(wl)&np.isfinite(d)&np.isfinite(e)&(e>0)&(wl>0.5e-6)
wl,d,e=wl[m],d[m],e[m]
edges=np.linspace(wl.min(), wl.max(), 50)
bc,bd,be=[],[],[]
for i in range(len(edges)-1):
    s=(wl>=edges[i])&(wl<edges[i+1])
    if s.sum()>=1:
        w=1/e[s]**2
        bc.append([edges[i],edges[i+1]]); bd.append((d[s]*w).sum()/w.sum()); be.append(1/np.sqrt(w.sum()))
bc,bd,be=np.array(bc),np.array(bd),np.array(be)
print("binned:",len(bc),"pts; depth range",bd.min(),bd.max())

guesses = {'Rs':1.0,'Mp':1.0,'Rp':1.0,'T':900.0,'logZ':0.0,'CO_ratio':0.53,'log_cloudtop_P':4.0,'log_k':None,'log_number_density':float('-inf'),'log_scatt_factor':0.0,'scatt_slope':4.0,'log_part_size':-6.0,'frac_scale_height':1.0,'profile_type':1.0,'n':None,'offset_transit':0.0,'offset_start':0.0,'offset_end':0.0,'offset_eclipse':0.0,'error_multiple':1.0,'T_star':5700.0,'T_spot':5700.0,'spot_cov_frac':0.0,'log_P_quench':-99.0,'log_CH4_mult':0.0,'fit_clr':False,'fit_vmr':False}
fi=FitInfo(guesses)
fi.add_uniform_fit_param("Rs",0.8,1.2,0.9,1.1)
fi.add_uniform_fit_param("Mp",0.5,2.0,0.9,1.1)
fi.add_uniform_fit_param("Rp",0.5,1.5,0.9,1.1)
fi.add_uniform_fit_param("T",500,1500,700,1100)
fi.add_uniform_fit_param("logZ",-2,3,-0.5,1.5)
fi.add_uniform_fit_param("CO_ratio",0.2,2.0,0.3,0.8)
fi.add_uniform_fit_param("log_cloudtop_P",2.0,6.0,2.5,5.5)
fi.add_gases_vmr(["H2O","CO2","CH4","CO","Na","K"], 1e-6, 1e-1)
print("fit params:", fi.fit_param_names)

calc=TransitDepthCalculator(); calc.change_wavelength_bins(bc)
ret=CombinedRetriever()
t0=time.time()
res=ret.run_emcee(bc,bd,be,None,None,None,fi,nwalkers=24,nsteps=400,include_condensation=False,rad_method="xsec")
el=time.time()-t0
samp=res.samples
iZ=fi.fit_param_names.index("logZ")
print("DONE in %.1fs, samples=%d"%(el,len(samp)))
print("logZ median=%.2f  IQR=%.2f (prior width=5.0)"%(np.median(samp[:,iZ]),np.percentile(samp[:,iZ],84)-np.percentile(samp[:,iZ],16)))
np.save(EXP+"/ck_stage6/smoke_Z.npy", samp)
