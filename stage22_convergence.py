"""Convergence diagnostics for the MCMC chains used in Figure 5 / Table 6.

Reports, for each retained run:
  * integrated autocorrelation time per parameter (emcee) and the implied
    effective sample size;
  * the ratio of chain length to autocorrelation time (the usual rule of thumb
    is >~50 independent samples per walker);
  * a split-chain consistency check: the median of the first half versus the
    second half, in units of the chain's own standard deviation, i.e. whether
    the two halves of each run agree with each other.

This is a diagnostic of the SAVED chains only; it does not change them.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CK = os.path.join(HERE, "ck_stage10")
PNAMES = ["Rp_m", "T_K", "logZ", "CO_ratio", "logPc"]   # column order of the saved chains
RUNS = ["IRonly_free", "optical_free"]

try:
    import emcee
except Exception:
    emcee = None


def autocorr(chain):
    """chain shape (nstep, nwalker, ndim) -> integrated time per dim."""
    if emcee is not None:
        try:
            return np.asarray(emcee.autocorr.integrated_time(chain, quiet=True))
        except Exception:
            pass
    # fallback: initial positive sequence estimator on the per-walker mean
    nstep, nwalk, ndim = chain.shape
    out = []
    for j in range(ndim):
        x = chain[:, :, j]
        x = x - x.mean(axis=0)
        var = (x ** 2).mean()
        if var <= 0:
            out.append(1.0)
            continue
        rho = []
        for lag in range(1, min(nstep // 4, 200)):
            c = (x[:-lag] * x[lag:]).mean() / var
            if c <= 0:
                break
            rho.append(c)
        tau = 1.0 + 2.0 * sum(rho)
        out.append(float(tau))
    return np.asarray(out)


def main():
    report = {}
    for lab in RUNS:
        f = os.path.join(CK, lab + "_chain.npy")
        if not os.path.exists(f):
            print("missing", f)
            continue
        ch = np.load(f)                      # (nstep, nwalker, ndim)
        nstep, nwalk, ndim = ch.shape
        tau = autocorr(ch)
        ess = nstep * nwalk / tau
        # split-chain check on the thinned (burn-in discarded) part
        post = ch[nstep // 2:]
        flat = post.reshape(-1, ndim)
        h1 = post[: post.shape[0] // 2].reshape(-1, ndim).mean(axis=0)
        h2 = post[post.shape[0] // 2:].reshape(-1, ndim).mean(axis=0)
        sd = flat.std(axis=0)
        wm_last = ch[-50:].mean(axis=0)          # per-walker mean of the last 50 steps
        wm_spread = (wm_last.max(axis=0) - wm_last.min(axis=0)) / sd
        nh = post[: post.shape[0] // 2].reshape(-1, ndim).shape[0]
        # difference of half-chain medians, scaled by the posterior sigma
        zsplit = np.abs(h1 - h2) / sd
        report[lab] = dict(nsteps=int(nstep), nwalkers=int(nwalk),
                           tau=[round(float(t), 1) for t in tau],
                           ess=[int(e) for e in ess],
                           steps_per_tau=[round(float(nstep / t), 1) for t in tau],
                           half_diff=[round(float(v), 4) for v in (h1 - h2)],
                           half_diff_in_sigma=[round(float(v), 3) for v in zsplit],
                           walker_mean_spread=[round(float(v), 4) for v in wm_spread])
        print("==", lab)
        print("   steps=%d walkers=%d" % (nstep, nwalk))
        for j, n in enumerate(PNAMES):
            print("   %-16s tau=%6.1f  ess=%6d  steps/tau=%6.1f  half1-half2=%+.4f  z=%.2f"
                  % (n, tau[j], ess[j], nstep / tau[j], (h1 - h2)[j], zsplit[j]))
    json.dump(report, open(os.path.join(HERE, "stage22_convergence.json"), "w",
                           encoding="utf-8"), indent=1)
    print("wrote stage22_convergence.json")


if __name__ == "__main__":
    main()
