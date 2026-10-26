"""
stage8_tp_profile.py  (NEW-1 robustness: T-P profile shape sensitivity, round 2)
=======================================================================
Clean isolation: the cloud-opacity horizon is about whether the cloud column
is optically thin at the photosphere. We test whether its location depends on
the T-P PROFILE SHAPE by fixing T (so the scalar-T column does not degenerate
the Jacobian under a custom profile) and varying ONLY the profile shape,
keeping the photospheric temperature ~constant. Free params = [logZ, CO_ratio,
log_cloudtop_P, Rp]; we scan log_cloudtop_P and locate the L=0.5 horizon.
Profiles (all sharing T_irr=1166 K at the column midpoint):
  isothermal, +400 K/dex (hotter deep), -400 K/dex (cooler deep), +800 K/dex.
Wavelength grid FIXED to real G395H. Horizon stable across shapes => the
horizon is an opacity/wavelength fact, not a thermal-assumption artifact.
"""
import csv, json, os, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identifiability import prior_leverage, whiten_jacobian, fisher

EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP, "stage8_tp_profile.json")
RS, MP, RP = 1.0*R_sun, 1.0*M_jup, 1.0*R_jup

rows = list(csv.DictReader(open(EXP+"/transitspec_jwst.csv", encoding="utf-8")))
grp = [r for r in rows if "WASP-39" in r["plntname"] and "ALDERSON" in r["plntranreflink"].upper()]
wl = np.array([float(r["centralwavelng"]) for r in grp])*1e-6
m = np.isfinite(wl)&(wl>0.5e-6); wl=wl[m]
edges = np.concatenate([[wl[0]], (wl[:-1]+wl[1:])/2, [wl[-1]]])
bins = np.column_stack([edges[:-1], edges[1:]])
sig = 0.0001*np.ones(len(bins))

T_irr = 1166.0
calc = TransitDepthCalculator(); calc.change_wavelength_bins(bins)
P_grid = np.array(calc.atm.P_grid, dtype=float)   # Pa, 13 layers
NL = len(P_grid)

def profile_with_gradient(grad, T_mid=T_irr):
    logP = np.log10(P_grid)
    T = T_mid + grad*(logP - 0.5*(logP[0]+logP[-1]))
    return P_grid.copy(), np.clip(T, 150.0, 3900.0)

def iso_profile():
    return P_grid.copy(), np.full(NL, T_irr)

profiles = {
    "isothermal":   iso_profile(),
    "grad_plus400": profile_with_gradient(+400.0),
    "grad_minus400":profile_with_gradient(-400.0),
    "grad_plus800": profile_with_gradient(+800.0),
}

theta_names = ["logZ","CO_ratio","log_cloudtop_P"]
S_a = np.diag([1.0, 0.25, 1.0])**2
logPc_grid = np.linspace(1.5, 6.5, 30)

def forward(logZ, CO, logPc, P_profile, T_profile):
    _, d, _ = calc.compute_depths(RS, MP, RP, None, logZ=logZ, CO_ratio=CO,
                                  cloudtop_pressure=10.0**logPc,
                                  custom_T_profile=T_profile, custom_P_profile=P_profile)
    return d

out = {}
for pname,(P_profile,T_profile) in profiles.items():
    hs=[]
    for logPc in logPc_grid:
        eps=np.array([0.05,0.02,0.05])
        f0=forward(0.0,0.53,logPc,P_profile,T_profile)
        J=np.zeros((len(bins),3))
        th=[0.0,0.53,logPc,RP]
        for j,name in enumerate(theta_names):
            th=[0.0,0.53,logPc]
            th[j]+=eps[j]; fp=forward(th[0],th[1],th[2],P_profile,T_profile); J[:,j]=(fp-f0)/eps[j]
        F=fisher(whiten_jacobian(J,sig,S_a))
        hs.append(prior_leverage(F)[2])  # index 2 = cloud-top pressure (0:logZ,1:CO,2:cloud)
    hv=np.array(hs); lg=logPc_grid
    if np.any((hv[:-1]-0.5)*(hv[1:]-0.5)<=0):
        k=np.where((hv[:-1]-0.5)*(hv[1:]-0.5)<=0)[0][0]
        t=(0.5-hv[k])/(hv[k+1]-hv[k]); horizon=lg[k]+t*(lg[k+1]-lg[k])
    else:
        horizon=float("nan")
    out[pname]={"horizon_logPc":horizon, "T_range":[float(T_profile.min()),float(T_profile.max())]}

json.dump(out, open(OUT,"w"), indent=1)
valid=[out[p]["horizon_logPc"] for p in out if out[p]["horizon_logPc"]==out[p]["horizon_logPc"]]
print("T-P profile robustness: horizons =", {p:round(out[p]["horizon_logPc"],3) for p in out})
if valid:
    print("spread across profiles: %.3f (min %.3f max %.3f)"%(np.ptp(valid),min(valid),max(valid)))
else:
    print("no crossing in 1.5-6.5 range for any profile; raw L at grid ends:",
          {p:[round(out[p]["horizon_logPc"],3) for p in out]})
