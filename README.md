# A Cloud-Opacity Horizon in JWST Transmission Retrievals

Analysis code and provenance ledger for the paper
*“A Cloud-Opacity Horizon in JWST Transmission Retrievals: What One Retrieval
Family Can and Cannot Measure”* (Tan Shuxin & Fu Sen, submitted to ApJ).

## What is here

* `stage1` – `stage22`: the analysis pipeline, in execution order. Each
  `stage*.py` writes its own `stage*.json` artifact next to it; every number in
  the paper is traceable to one of those files.
* `identifiability.py`: the averaging-kernel / Fisher / bias-transfer operators.
* `figstyle.py`, `make_figures.py`: figure styling and figure assembly.
* `regen_mcmc_numbers.py`: rewrites the manuscript's MCMC-dependent numbers
  directly from the artifacts (so text and artifacts cannot drift apart).
* `provenance.md`: the ledger mapping each reported number to its artifact.
* `transitspec_jwst.csv`, `system_params.csv`: the NASA Exoplanet Archive input
  used (no simulated data anywhere in the pipeline).

Machine-readable results (`summary.csv`, `horizon_curves.csv`, `README.txt`) are
published with the article itself.

## Environment

Python 3.13; `numpy`, `scipy`, `matplotlib`, `emcee`, `corner`, and PLATON
(M. Zhang et al. 2020). Run with `OMP_NUM_THREADS=1` (the scripts set this
internally) and note that the radiative transfer is the expensive step: the
cloud-horizon bisection needs ~15–20 forward-model evaluations per spectrum, and
the 3000-step emcee chains take ~1 h each on one core.

## Reproducing the headline numbers

```bash
python stage2_real_diagnostics.py     # per-channel errors, Jacobians, d_s
python stage4_horizon_and_bias.py     # horizon (L=0.5) + linear bias transfer
python stage5_nonlinear_check.py      # nonlinear refit vs linear prediction
python stage20_threshold.py           # horizon at L = 0.25 / 0.50 / 0.75
python stage21_coverage.py            # controlled coverage test on one spectrum
python stage10_retrieval_long.py      # emcee retrievals (set PLATON_NSTEPS=3000)
python stage22_convergence.py         # autocorrelation / half-chain diagnostics
python stage18_posteriors.py          # posterior summaries and corner figures
```