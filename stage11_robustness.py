"""
stage11_robustness.py  --  REVISION ROUND 1 (referee #15, #21, #22, #23)
=======================================================================
Recomputes the two robustness experiments of Section 6 with the SAME
conventions as the main diagnostic, which stage7/stage8 did not:

  stage7 / stage8 used  Rs = 1 R_sun, Mp = 1 M_jup, Rp = 1 R_jup  (a generic
  Jupiter, not WASP-39 b), a synthetic flat sigma = 100 ppm for every channel,
  and a different prior width (200 K, 0.25, 1.0 dex) from the one the main
  results actually used (150 K, 0.30, 1.5 dex -- stage2 PRIOR_STD).

Everything here uses:
  * the real system parameters of the planet (Table 3),
  * the real published per-channel sigma_i of the real dataset,
  * the same prior widths and finite-difference steps as Sections 3-5
    (imported from stage2_real_diagnostics).

Parts:
  A  fiducial-atmosphere robustness (Section 6.5)
  B  T-P profile shape robustness (Section 6.4) -- incl. the standard
     5-parameter isothermal recomputation as a consistency check against
     Section 4, and the profiled campaign of tested shapes for the figure
  C  mean-molecular-weight / metallicity robustness for the high-MMW planet
     in the sample, GJ 486 b (Section 6.6 caveat, referee #1 and #23)

Nothing is simulated: all sigma come from the published spectra.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from platon.transit_depth_calculator import TransitDepthCalculator
from stage2_real_diagnostics import (load_systems, load_datasets,
                                     bins_from_centres, PNAMES, FD, PRIOR_STD)

OUTA = os.path.join(HERE, "stage11_fiducial.json")
OUTB = os.path.join(HERE, "stage11_tp.json")
OUTC = os.path.join(HERE, "stage11_mmw.json")

LOGPC = np.arange(1.5, 6.51, 0.25)
PSTD5 = np.array([PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
                  PRIOR_STD["log_cloudtop_P"]])


def get_dataset(planet_sub, ref_sub):
    for d in load_datasets():
        if planet_sub in d["planet"] and ref_sub in d["ref"].upper():
            return d
    raise KeyError((planet_sub, ref_sub))


def make_calc(ds):
    calc = TransitDepthCalculator()
    calc.change_wavelength_bins(bins_from_centres(ds["wl_um"]))
    return calc


def pstd(theta0, drop_T=False):
    Rp_term = theta0[3] if drop_T else theta0[4]
    base = [PRIOR_STD["T"], PRIOR_STD["logZ"], PRIOR_STD["CO_ratio"],
            PRIOR_STD["log_cloudtop_P"], 0.01 * Rp_term]
    if drop_T:
        base = base[1:]
    return np.array(base)


def jac(fwd, th0, fd=None, Rp_step=0.002):
    """Central-difference Jacobian.  Returns None if any forward evaluation
    raises (PLATON raises AtmosphereError for unphysical combinations such as
    a very hot deep atmosphere) so callers can record the point as missing."""
    try:
        f0 = np.asarray(fwd(th0), float)
        if not np.all(np.isfinite(f0)):
            return None, None
        n = len(th0)
        steps = np.array([FD["T"], FD["logZ"], FD["CO_ratio"], FD["log_cloudtop_P"],
                          Rp_step])[:n] if fd is None else fd
        J = np.zeros((f0.size, n))
        for j in range(n):
            tp = np.array(th0, float); tp[j] += steps[j]
            tm = np.array(th0, float); tm[j] -= steps[j]
            fp = np.asarray(fwd(tp), float); fm = np.asarray(fwd(tm), float)
            if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fm))):
                return None, None
            J[:, j] = (fp - fm) / (2 * steps[j])
        return J, f0
    except Exception:
        return None, None


def cloud_leverage(J, sigma, S_a, idx_cloud):
    import identifiability as idf
    F = idf.fisher(idf.whiten_jacobian(J, sigma, S_a))
    return float(idf.prior_leverage(F)[idx_cloud]), F


def horizon_from_scan(values, grid):
    hv = np.asarray(values, float)
    ok = np.isfinite(hv)
    if ok.sum() < 2:
        return float("nan")
    idx = np.where(ok)[0]
    hv, g = hv[idx], np.asarray(grid, float)[idx]
    sgn = (hv[:-1] - 0.5) * (hv[1:] - 0.5)
    if np.any(sgn <= 0):
        k = int(np.where(sgn <= 0)[0][0])
        t = (0.5 - hv[k]) / (hv[k + 1] - hv[k])
        return float(g[k] + t * (g[k + 1] - g[k]))
    return float("nan")


def column_corr(J, sigma, i, j):
    """cosine similarity of two whitened Jacobian columns: how degenerate
    the two parameters are at this fiducial state."""
    Kn = J / sigma[:, None]
    a, b = Kn[:, i], Kn[:, j]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


# ----------------------------------------------------------------- Part A
def partA():
    sy = load_systems()["WASP-39 b"]
    ds = get_dataset("WASP-39", "ALDERSON")
    calc = make_calc(ds)
    sigma = ds["sigma"]
    Rp0 = sy["Rp"] / 7.1492e7

    def fwd(th):
        _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[4] * 7.1492e7, th[0],
                                      logZ=th[1], CO_ratio=th[2],
                                      cloudtop_pressure=10.0 ** th[3])
        return np.asarray(d, float)

    rows = []
    grid = [(600.0, 1.0, 0.53), (852.0, 0.0, 0.35), (852.0, 0.0, 0.70),
            (852.0, 1.0, 0.35), (852.0, 1.0, 0.70), (852.0, 2.0, 0.35),
            (852.0, 2.0, 0.70), (1166.0, 1.0, 0.53), (1166.0, 0.0, 0.35), (1166.0, 0.0, 0.70), (1166.0, 1.0, 0.35),
            (1166.0, 1.0, 0.70), (1166.0, 2.0, 0.35), (1166.0, 2.0, 0.70)]
    for (T, lz, co) in grid:
        th0 = np.array([T, lz, co, 4.0, Rp0])
        Sa = pstd(th0) ** 2
        vals, cors = [], []
        t0 = time.time()
        for lpc in LOGPC:
            th = np.array([T, lz, co, float(lpc), Rp0])
            J, _ = jac(fwd, th)
            if J is None:
                vals.append(float("nan"))
                continue
            L, _ = cloud_leverage(J, sigma, Sa, 3)
            vals.append(L)
            if abs(lpc - 4.0) < 1e-9:
                cors = dict(logZ_cloud=column_corr(J, sigma, 1, 3),
                            CO_cloud=column_corr(J, sigma, 2, 3),
                            T_cloud=column_corr(J, sigma, 0, 3))
        h = horizon_from_scan(vals, LOGPC)
        rows.append(dict(fiducial=dict(T=T, logZ=lz, CO=co), horizon_logPc=h,
                         L_at_grid_end=float(vals[-1]), L_first=float(vals[0]),
                         column_corr_at_4dex=cors, seconds=round(time.time() - t0, 1)))
        print("  fid T=%4.0f logZ=%.1f CO=%.2f -> horizon=%s  %.0fs"
              % (T, lz, co, ("%.3f" % h) if h == h else "none", time.time() - t0),
              flush=True)
        json.dump(rows, open(OUTA, "w", encoding="utf-8"), indent=1)
    v = np.array([r["horizon_logPc"] for r in rows], float)
    vv = v[np.isfinite(v)]
    summary = dict(per_fiducial=rows, n_valid=int(vv.size),
                   mean=float(vv.mean()) if vv.size else None,
                   std=float(vv.std()) if vv.size else None,
                   min=float(vv.min()) if vv.size else None,
                   max=float(vv.max()) if vv.size else None,
                   note="real WASP-39 b system parameters, real per-channel sigma, "
                        "prior = stage2 PRIOR_STD (150 K, 1.0 dex, 0.30, 1.5 dex, 1% Rp)")
    json.dump(summary, open(OUTA, "w", encoding="utf-8"), indent=1)
    if vv.size:
        print("A: n_valid=%d  mean=%.3f  std=%.3f  range=[%.3f, %.3f]"
              % (vv.size, vv.mean(), vv.std(), vv.min(), vv.max()), flush=True)


# ----------------------------------------------------------------- Part B
def profiles(calc, T_irr):
    """Alternative T-P shapes sharing the same photospheric temperature.
    Gradients are linear in log10 P and clipped to +/-40% of T_irr, which
    brackets the shape range spanned by self-consistent hot-Jupiter profiles
    in the transmission photosphere (an unclipped several-thousand-K excursion
    makes the atmosphere formally unbound and PLATON refuses the model)."""
    P = np.array(calc.atm.P_grid, dtype=float)
    logP = np.log10(P)
    c = 0.5 * (logP[0] + logP[-1])
    lo, hi = 0.7 * T_irr, 1.4 * T_irr
    out = {"isothermal": np.full(len(P), T_irr)}
    for g in (+200.0, -200.0, +400.0, -400.0):
        Tg = np.clip(T_irr + g * (logP - c), max(150.0, lo), min(3900.0, hi))
        out["grad%+dk_per_dex" % int(g)] = Tg
    return P, out


def partB():
    sy = load_systems()["WASP-39 b"]
    ds = get_dataset("WASP-39", "ALDERSON")
    calc = make_calc(ds)
    sigma = ds["sigma"]
    Rp0 = sy["Rp"] / 7.1492e7
    Tirr = float(sy["Teq"])

    # B1: standard 5-parameter isothermal route (Section 4 method) as a check
    def fwd_iso(th):
        _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[4] * 7.1492e7, th[0],
                                      logZ=th[1], CO_ratio=th[2],
                                      cloudtop_pressure=10.0 ** th[3])
        return np.asarray(d, float)

    th_ref = np.array([Tirr, 1.0, 0.53, 4.0, Rp0])
    Sa5 = pstd(th_ref) ** 2
    vals = []
    for lpc in LOGPC:
        th = np.array([Tirr, 1.0, 0.53, float(lpc), Rp0])
        J, _ = jac(fwd_iso, th)
        if J is None:
            vals.append(float("nan")); continue
        L, _ = cloud_leverage(J, sigma, Sa5, 3)
        vals.append(L)
    h_iso = horizon_from_scan(vals, LOGPC)
    print("  B1 standard isothermal (5 par, Section-4 method): %s"
          % (("horizon=%.3f" % h_iso) if h_iso == h_iso else "no crossing"), flush=True)

    # B2: custom profiles (T fixed -> 4 free parameters)
    P_grid, profs = profiles(calc, Tirr)
    rows = []
    for name, Tprof in profs.items():
        def fwd(th, Tprof=Tprof):
            _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[3] * 7.1492e7, None,
                                          logZ=th[0], CO_ratio=th[1],
                                          cloudtop_pressure=10.0 ** th[2],
                                          custom_T_profile=Tprof, custom_P_profile=P_grid)
            return np.asarray(d, float)

        thref = np.array([1.0, 0.53, 4.0, Rp0])
        steps = np.array([FD["logZ"], FD["CO_ratio"], FD["log_cloudtop_P"], 0.002])
        Sa4 = pstd(thref, drop_T=True) ** 2
        vals, nfail = [], 0
        for lpc in LOGPC:
            th = np.array([1.0, 0.53, float(lpc), Rp0])
            J, _ = jac(fwd, th, fd=steps)
            if J is None:
                vals.append(float("nan")); nfail += 1; continue
            L, _ = cloud_leverage(J, sigma, Sa4, 2)
            vals.append(L)
        h = horizon_from_scan(vals, LOGPC)
        rows.append(dict(profile=name, horizon_logPc=h, n_grid_points_failed=nfail,
                         T_min=float(np.min(Tprof)), T_max=float(np.max(Tprof)),
                         T_profile=[float(x) for x in Tprof],
                         L_first=float(vals[0]), L_last=float(vals[-1]),
                         L_curve=[float(x) for x in vals]))
        print("  B2 %-22s horizon=%s  L(1.5)=%.3f L(6.5)=%.3f  T range %.0f-%.0f K"
              % (name, ("%.3f" % h) if h == h else "none", vals[0], vals[-1],
                 Tprof.min(), Tprof.max()), flush=True)
        json.dump(dict(h_standard_isothermal=h_iso, profiles=rows,
                       P_grid=[float(x) for x in P_grid],
                       logPc_grid=[float(x) for x in LOGPC]),
                  open(OUTB, "w", encoding="utf-8"), indent=1)
    json.dump(dict(h_standard_isothermal=h_iso, profiles=rows,
                   P_grid=[float(x) for x in P_grid],
                   logPc_grid=[float(x) for x in LOGPC]),
              open(OUTB, "w", encoding="utf-8"), indent=1)


# ----------------------------------------------------------------- Part C
def partC():
    sy = load_systems()["GJ 486 b"]
    ds_list = [d for d in load_datasets() if "GJ 486" in d["planet"]]
    ds = max(ds_list, key=lambda d: len(d["wl_um"]))
    calc = make_calc(ds)
    sigma = ds["sigma"]
    Rp0 = sy["Rp"] / 7.1492e7
    rows = []
    for lz in (0.0, 1.0, 2.0, 3.0):
        def fwd(th):
            _, d, _ = calc.compute_depths(sy["Rs"], sy["Mp"], th[4] * 7.1492e7, th[0],
                                          logZ=th[1], CO_ratio=th[2],
                                          cloudtop_pressure=10.0 ** th[3])
            return np.asarray(d, float)

        # mean molecular weight at the fiducial state
        _, _, info = calc.compute_depths(sy["Rs"], sy["Mp"], Rp0 * 7.1492e7,
                                         float(sy["Teq"]), logZ=lz, CO_ratio=0.53,
                                         cloudtop_pressure=1e4, full_output=True)
        mu = info.get("mu_profile")
        mu_mean = float(np.mean(mu)) if mu is not None else None
        thref = np.array([float(sy["Teq"]), lz, 0.53, 4.0, Rp0])
        Sa = pstd(thref) ** 2
        vals = []
        for lpc in LOGPC:
            th = np.array([float(sy["Teq"]), lz, 0.53, float(lpc), Rp0])
            J, _ = jac(fwd, th)
            if J is None:
                vals.append(float("nan")); continue
            L, _ = cloud_leverage(J, sigma, Sa, 3)
            vals.append(L)
        h = horizon_from_scan(vals, LOGPC)
        rows.append(dict(logZ=lz, mu_mean_amu=mu_mean, horizon_logPc=h,
                         L_first=float(vals[0]), L_last=float(vals[-1]),
                         L_curve=[float(x) for x in vals]))
        print("  C GJ486b logZ=%.1f  mu=%.2f amu  horizon=%s  L(1.5)=%.3f L(6.5)=%.3f"
              % (lz, mu_mean if mu_mean else -1, ("%.3f" % h) if h == h else "none",
                 vals[0], vals[-1]), flush=True)
        json.dump(rows, open(OUTC, "w", encoding="utf-8"), indent=1)
    json.dump(dict(n_channels=int(len(sigma)), rows=rows,
                   logPc_grid=[float(x) for x in LOGPC]),
              open(OUTC, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["A", "B", "C"]
    if "A" in parts:
        print("PART A: fiducial robustness (real WASP-39 b)", flush=True); partA()
    if "B" in parts:
        print("PART B: T-P profile robustness", flush=True); partB()
    if "C" in parts:
        print("PART C: metallicity / mean molecular weight (GJ 486 b)", flush=True); partC()
