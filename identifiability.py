"""
identifiability.py
==================
Core diagnostic machinery of the paper. Pure functions, no I/O, no data.

Everything here is standard optimal-estimation / information-geometry theory
(Rodgers 2000) adapted to the transmission-spectroscopy forward operator.
The novelty of the paper is not any single formula below, but (a) applying the
averaging-kernel / DOFS construction to exoplanet transmission retrievals,
(b) the prior-leverage decomposition L(theta), and (c) tying both to
distribution-free coverage calibration.

Definitions
-----------
Forward model:      d = F(theta) + eps,  eps ~ N(0, S_eps), S_eps diagonal
Jacobian:           K_ij = dF_i / dtheta_j
Prior:              theta ~ N(theta_a, S_a)   (S_a encodes the physical prior)

Whitened Jacobian:  Ktilde = S_eps^{-1/2} K S_a^{1/2}       (dimensionless)
Fisher matrix:      Fisher = Ktilde^T Ktilde
Posterior (linear): Shat_scaled = (I + Fisher)^{-1}         (in prior units)
Averaging kernel:   A = (I + Fisher)^{-1} Fisher = I - Shat_scaled
DOFS:               d_s = tr(A) = sum_i lam_i/(1+lam_i),  lam = eig(Fisher)

Prior leverage (this paper):
    L_j = 1 - A_jj  in [0, 1]
i.e. the fraction of the marginal posterior variance of theta_j that is
supplied by the prior rather than by the data. L_j -> 1 means "the number you
report for theta_j is your prior, not your measurement".

Information-limited resolution:
    the eigenvectors of Fisher with lam < 1 are directions the data cannot
    distinguish from the prior; we call span{v : lam(v) < 1} the
    *prior-dominated subspace* and report its dimension and composition.
"""
import numpy as np

__all__ = [
    "whiten_jacobian", "fisher", "averaging_kernel", "dofs",
    "prior_leverage", "prior_dominated_subspace", "posterior_covariance",
    "degeneracy_directions", "bias_from_fixing", "gauss_newton_step",
    "effective_rank", "resolution_matrix_summary",
]


def _sqrtm_diag(S):
    """Matrix square root of a covariance given as 1-D variances or 2-D matrix."""
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        return np.diag(np.sqrt(S))
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def _invsqrtm_diag(S):
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        return np.diag(1.0 / np.sqrt(S))
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-300, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def whiten_jacobian(K, sigma, S_a):
    """Ktilde = S_eps^{-1/2} K S_a^{1/2}.

    Parameters
    ----------
    K : (m, n) Jacobian dF/dtheta in physical units
    sigma : (m,) per-channel 1-sigma measurement uncertainties
    S_a : (n,) prior variances or (n, n) prior covariance
    """
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float).ravel()
    assert K.shape[0] == sigma.size, (K.shape, sigma.shape)
    Sa_half = _sqrtm_diag(S_a)
    assert Sa_half.shape[0] == K.shape[1], (K.shape, Sa_half.shape)
    return (K / sigma[:, None]) @ Sa_half


def fisher(Ktilde):
    """Dimensionless Fisher information in prior units."""
    return Ktilde.T @ Ktilde


def averaging_kernel(F):
    """A = (I + F)^{-1} F. Rows describe how retrieved parameters average truth."""
    n = F.shape[0]
    I = np.eye(n)
    return np.linalg.solve(I + F, F)


def posterior_covariance(F, S_a):
    """Posterior covariance in PHYSICAL units: S_a^{1/2}(I+F)^{-1}S_a^{1/2}."""
    n = F.shape[0]
    Sa_half = _sqrtm_diag(S_a)
    return Sa_half @ np.linalg.inv(np.eye(n) + F) @ Sa_half


def dofs(F):
    """Degrees of freedom for signal, tr(A) = sum lam/(1+lam)."""
    lam = np.linalg.eigvalsh(F)
    lam = np.clip(lam, 0.0, None)
    return float(np.sum(lam / (1.0 + lam)))


def prior_leverage(F):
    """L_j = 1 - A_jj: fraction of parameter j's constraint coming from prior."""
    A = averaging_kernel(F)
    return 1.0 - np.diag(A)


def effective_rank(F, thresh=1.0):
    """Number of Fisher eigenvalues above `thresh` (data beats prior)."""
    lam = np.linalg.eigvalsh(F)
    return int(np.sum(lam > thresh))


def prior_dominated_subspace(F, thresh=1.0):
    """Eigen-decomposition split into data-dominated / prior-dominated parts.

    Returns dict with eigenvalues (descending), eigenvectors as columns, and
    the index at which lam drops below `thresh`.
    """
    lam, V = np.linalg.eigh(F)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    k = int(np.sum(lam > thresh))
    return dict(eigenvalues=lam, eigenvectors=V, n_data_dominated=k,
                n_prior_dominated=int(F.shape[0] - k))


def degeneracy_directions(F, top=2):
    """The `top` least-constrained directions (smallest Fisher eigenvalues).

    These are the concrete degeneracy axes, e.g. a mixture of cloud-top
    pressure and metallicity, expressed in prior-normalised parameter units.
    """
    lam, V = np.linalg.eigh(F)
    order = np.argsort(lam)
    return [(float(lam[i]), V[:, i]) for i in order[:top]]


def bias_from_fixing(K, sigma, S_a, idx_fixed, delta):
    """Bias induced in the free parameters by fixing theta[idx_fixed] wrong.

    This is the *cost* side of the identifiability/bias trade-off: if a physical
    prior pins a parameter at the wrong value by `delta`, the retrieved free
    parameters absorb it as

        bias_free = -(Kf^T Se^-1 Kf + Sa_ff^-1)^{-1} Kf^T Se^-1 Kx delta

    Derivation: with d = F(theta_true) and the fixed parameter offset by delta,
    the weighted residual is r = -Kx delta - Kf dtheta_f; the MAP normal
    equations give the expression above. The leading minus sign matters and was
    verified against a fully nonlinear Gauss-Newton refit (stage5).

    Returns (free_indices, bias) with bias in physical units of the free
    parameters.
    """
    K = np.asarray(K, float)
    sigma = np.asarray(sigma, float).ravel()
    n = K.shape[1]
    idx_fixed = np.atleast_1d(idx_fixed)
    free = np.array([j for j in range(n) if j not in set(idx_fixed.tolist())])
    Kf = K[:, free] / sigma[:, None]
    Kx = K[:, idx_fixed] / sigma[:, None]
    Sa = np.asarray(S_a, float)
    Sa_ff_inv = np.diag(1.0 / Sa[free]) if Sa.ndim == 1 else np.linalg.inv(Sa[np.ix_(free, free)])
    M = Kf.T @ Kf + Sa_ff_inv
    rhs = Kf.T @ (Kx @ np.atleast_1d(delta))
    return free, -np.linalg.solve(M, rhs)


def gauss_newton_step(K, sigma, S_a, theta, theta_a, residual, damping=1.0):
    """One regularised Gauss-Newton / Levenberg-Marquardt step (for MAP)."""
    K = np.asarray(K, float)
    sigma = np.asarray(sigma, float).ravel()
    Kn = K / sigma[:, None]
    rn = np.asarray(residual, float).ravel() / sigma
    Sa = np.asarray(S_a, float)
    Sa_inv = np.diag(1.0 / Sa) if Sa.ndim == 1 else np.linalg.inv(Sa)
    H = Kn.T @ Kn + damping * Sa_inv
    g = Kn.T @ rn - Sa_inv @ (np.asarray(theta) - np.asarray(theta_a))
    return np.linalg.solve(H, g)


def resolution_matrix_summary(F, names):
    """Human-readable audit of what is and is not measured."""
    A = averaging_kernel(F)
    L = 1.0 - np.diag(A)
    sub = prior_dominated_subspace(F)
    rows = []
    for j, nm in enumerate(names):
        rows.append(dict(parameter=nm, A_diag=float(A[j, j]),
                         prior_leverage=float(L[j]),
                         verdict=("measured" if A[j, j] > 0.5 else
                                  "prior-dominated" if A[j, j] < 0.2 else "partial")))
    return dict(dofs=dofs(F), n_params=len(names),
                n_data_dominated=sub["n_data_dominated"],
                eigenvalues=[float(x) for x in sub["eigenvalues"]],
                per_parameter=rows)
