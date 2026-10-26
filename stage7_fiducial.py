"""
stage7_fiducial_robustness.py  (corrected experimental design)
===========================================================
ADVERSIAL SELF-REVIEW ITEM #2: is the cloud-opacity horizon an
opacity/geometry fact (robust to the chosen fiducial atmosphere) or a
definition artifact of the fiducial guess?

Correct design: FIX the wavelength grid + noise to ONE real IR-only
dataset (Alderson G395H, 344 channels, 2.9-5.2 um). Vary ONLY the
fiducial atmosphere (T, logZ, CO) at which the Jacobian is evaluated,
and scan cloud-top pressure to locate the horizon. If the horizon is
stable across fiducials, it is a property of the opacity structure, not
the guess. If it moves a lot, we soften the claim.
"""
import csv, json, os, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from platon.transit_depth_calculator import TransitDepthCalculator
from platon.constants import R_sun, R_jup, M_jup
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identifiability import prior_leverage, whiten_jacobian, fisher

EXP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(EXP, "stage7_fiducial.json")
CK = os.path.join(EXP, "ck_stage7"); os.makedirs(CK, exist_ok=True)
RS, MP, RP = 1.0*R_sun, 1.0*M_jup, 1.0*R_jup

# --- load Alderson G395H real grid (fixed throughout) ---
rows = list(csv.DictReader(open(EXP+"/transitspec_jwst.csv", encoding="utf-8")))
grp = [r for r in rows if "WASP-39" in r["plntname"] and "ALDERSON" in r["plntranreflink"].upper()]
wl = np.array([float(r["centralwavelng"]) for r in grp])*1e-6  # metres
m = np.isfinite(wl) & (wl>0.5e-6)
wl = np.sort(wl[m])
edges = np.concatenate([[wl[0]], (wl[:-1]+wl[1:])/2, [wl[-1]]])
bins = np.column_stack([edges[:-1], edges[1:]])
sig = 0.0001*np.ones(len(bins))  # 100 ppm representative, held FIXED

theta_names = ["T","logZ","CO_ratio","log_cloudtop_P","Rp"]
S_a = np.diag([200.0, 1.0, 0.25, 1.0, 0.01*RP])**2
logPc_grid = np.linspace(2.0, 5.5, 20)

calc = TransitDepthCalculator(); calc.change_wavelength_bins(bins)
def forward(T,logZ,CO,logPc,Rp=RP):
    _, d, _ = calc.compute_depths(RS, MP, Rp, T, logZ=logZ, CO_ratio=CO,
                                  cloudtop_pressure=10.0**logPc)
    return d

def horizon_at_fiducial(T, logZ, CO):
    hs=[]
    for logPc in logPc_grid:
        eps=np.array([30.0,0.05,0.02,0.05,0.01*RP])
        f0=forward(T,logZ,CO,logPc)
        J=np.zeros((len(bins),5))
        for j,name in enumerate(theta_names):
            th=[T,logZ,CO,logPc,RP]
            if name=="Rp":
                fp=forward(T,logZ,CO,logPc,RP*(1+1e-2)); J[:,j]=(fp-f0)/(1e-2*RP)
            else:
                th[j]+=eps[j]; fp=forward(*th); J[:,j]=(fp-f0)/eps[j]
        F=fisher(whiten_jacobian(J,sig,S_a))
        hs.append(prior_leverage(F)[3])
    hv=np.array(hs); lg=logPc_grid
    if np.any((hv[:-1]-0.5)*(hv[1:]-0.5)<=0):
        k=np.where((hv[:-1]-0.5)*(hv[1:]-0.5)<=0)[0][0]
        t=(0.5-hv[k])/(hv[k+1]-hv[k]); return lg[k]+t*(lg[k+1]-lg[k])
    return float("nan")

grid=[(600,0.0,0.53),(600,1.0,0.53),(600,2.0,0.53),
      (852,0.0,0.53),(852,1.0,0.53),(852,2.0,0.53),
      (1100,0.0,0.53),(1100,1.0,0.53),(1100,2.0,0.53),
      (852,1.0,0.35),(852,1.0,0.70)]
rows_out=[]
for (T,lZ,CO) in grid:
    h=horizon_at_fiducial(T,lZ,CO)
    rows_out.append({"fiducial":{"T":T,"logZ":lZ,"CO":CO},"horizon_logPc":h})
hs=np.array([r["horizon_logPc"] for r in rows_out])
valid=hs[np.isfinite(hs)]
summary={"per_fiducial":rows_out,
         "n_valid":int(len(valid)),
         "mean_horizon":float(valid.mean()) if len(valid) else None,
         "std_horizon":float(valid.std()) if len(valid) else None,
         "min":float(valid.min()) if len(valid) else None,
         "max":float(valid.max()) if len(valid) else None,
         "note":"wavelength grid FIXED to Alderson G395H; only fiducial atmosphere varied"}
json.dump(summary, open(OUT,"w"), indent=1)
print("FIXED-GRID fiducial scan: valid=%d mean=%.3f std=%.3f range=[%.3f,%.3f]"%(
    len(valid), valid.mean(), valid.std(), valid.min(), valid.max()))
