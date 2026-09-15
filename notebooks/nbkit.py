"""Shared utilities for the exposome-modelling notebooks.

Deliberately small and dependency-light (numpy + matplotlib). Each notebook implements
the *new* idea of its Part from scratch; this module only holds the pieces that would
otherwise be retyped every time -- basic kernels, PSD diagnostics, plot styling, and the
synthetic exposome generator used from Part 3 onwards.

See notebooks/README.md for the roadmap.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "set_style", "PALETTE",
    "linear_kernel", "poly_kernel", "rbf_kernel", "ard_rbf_kernel", "matern52_kernel",
    "periodic_kernel", "white_kernel",
    "gram", "is_psd", "psd_report", "nearest_psd", "jitter_chol",
    "center_kernel", "kernel_alignment",
    "make_exposome", "standardize",
]

PALETTE = {
    "pos": "#3b6ea5",
    "neg": "#c46a4a",
    "accent": "#6a8f6b",
    "muted": "#8a8a8a",
    "band": "#3b6ea5",
    "cycle": ["#3b6ea5", "#c46a4a", "#6a8f6b", "#9a6fa5", "#c9a227", "#4f9aa8"],
}


def set_style(figsize=(5.0, 4.0), dpi=120):
    """Consistent, low-chrome matplotlib defaults across all notebooks."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE["cycle"]),
        "figure.constrained_layout.use": False,
    })
    return plt


# --------------------------------------------------------------------------------------
# Kernels.  All take (A, B) with shape (n, d), (m, d) and return the (n, m) Gram matrix.
# --------------------------------------------------------------------------------------

def _sqdist(A, B):
    """Squared Euclidean distances, clipped at 0 to kill round-off negatives."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)


def linear_kernel(A, B, variance=1.0, offset=0.0):
    return variance * (np.atleast_2d(A) @ np.atleast_2d(B).T) + offset


def poly_kernel(A, B, degree=3, coef0=1.0, gamma=1.0):
    return (gamma * (np.atleast_2d(A) @ np.atleast_2d(B).T) + coef0) ** degree


def rbf_kernel(A, B, lengthscale=1.0, variance=1.0):
    """Squared-exponential / Gaussian kernel, parameterised by lengthscale (not gamma)."""
    return variance * np.exp(-0.5 * _sqdist(A, B) / np.asarray(lengthscale) ** 2)


def ard_rbf_kernel(A, B, lengthscales, variance=1.0):
    """Automatic Relevance Determination: one lengthscale per input dimension.

    Large lengthscale_j => input j is (nearly) irrelevant. This is the workhorse of
    Parts 2 and 5.
    """
    ell = np.atleast_1d(np.asarray(lengthscales, dtype=float))
    A = np.atleast_2d(A) / ell
    B = np.atleast_2d(B) / ell
    return variance * np.exp(-0.5 * _sqdist(A, B))


def matern52_kernel(A, B, lengthscale=1.0, variance=1.0):
    r = np.sqrt(_sqdist(A, B)) / lengthscale
    s5 = np.sqrt(5.0)
    return variance * (1.0 + s5 * r + 5.0 / 3.0 * r**2) * np.exp(-s5 * r)


def periodic_kernel(A, B, period=1.0, lengthscale=1.0, variance=1.0):
    """MacKay's periodic kernel, applied per dimension and multiplied.

    The per-dimension product matters: summing |x_j - x'_j| inside the sine first and
    then exponentiating is *not* positive definite. Products of valid kernels are valid
    (Part 1 sec 1.4 closure rules), so build it that way.
    """
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    d = np.abs(A[:, None, :] - B[None, :, :])                       # (n, m, d)
    logk = -2.0 * np.sin(np.pi * d / period) ** 2 / lengthscale**2
    return variance * np.exp(logk.sum(-1))                          # product over dims


def white_kernel(A, B, variance=1.0):
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    if A.shape == B.shape and np.allclose(A, B):
        return variance * np.eye(len(A))
    return np.zeros((len(A), len(B)))


def gram(kernel, X, Z=None, **kw):
    """K = kernel(X, Z or X), symmetrised when it is a self-Gram."""
    if Z is None:
        K = kernel(X, X, **kw)
        return 0.5 * (K + K.T)
    return kernel(X, Z, **kw)


# --------------------------------------------------------------------------------------
# Positive-definiteness diagnostics (Part 1 sec 1.4, reused whenever a kernel is invented)
# --------------------------------------------------------------------------------------

def is_psd(K, tol=1e-9):
    K = 0.5 * (np.asarray(K) + np.asarray(K).T)
    ev = np.linalg.eigvalsh(K)
    scale = max(1.0, np.abs(ev).max())
    return bool(ev.min() > -tol * scale)


def psd_report(name, K, tol=1e-9, verbose=True):
    """Print lambda_min and a verdict; return the eigenvalues."""
    K = 0.5 * (np.asarray(K) + np.asarray(K).T)
    ev = np.linalg.eigvalsh(K)
    ok = is_psd(K, tol)
    if verbose:
        rank = int((ev > 1e-10 * max(1.0, ev.max())).sum())
        print(f"{name:<36} lambda_min = {ev.min():+.3e}   rank {rank:>3}/{len(ev)}   "
              f"{'PSD  ok' if ok else 'NOT PSD'}")
    return ev


def nearest_psd(K):
    """Project a symmetric matrix onto the PSD cone by clipping negative eigenvalues."""
    K = 0.5 * (np.asarray(K) + np.asarray(K).T)
    w, V = np.linalg.eigh(K)
    return (V * np.maximum(w, 0.0)) @ V.T


def jitter_chol(K, jitter=1e-8, max_tries=8):
    """Cholesky with escalating jitter. Returns (L, jitter_used).

    Every GP routine in these notebooks goes through here -- adding a little to the
    diagonal is the standard fix for a Gram matrix that is PSD in theory but indefinite
    in float64.
    """
    K = 0.5 * (np.asarray(K, dtype=float) + np.asarray(K, dtype=float).T)
    n = len(K)
    scale = np.mean(np.diag(K))
    for i in range(max_tries):
        j = jitter * (10.0**i)
        try:
            return np.linalg.cholesky(K + j * scale * np.eye(n)), j * scale
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky failed even with maximum jitter")


def center_kernel(K, K_test=None):
    """Centre a Gram matrix in feature space: Ktilde = H K H, H = I - 11'/n (kernel PCA)."""
    K = np.asarray(K)
    n = len(K)
    one_n = np.ones((n, n)) / n
    if K_test is None:
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n
    m = len(K_test)
    one_m = np.ones((m, n)) / n
    return K_test - one_m @ K - K_test @ one_n + one_m @ K @ one_n


def kernel_alignment(K1, K2):
    """Centred kernel alignment in [0, 1] -- how similar two kernels' geometries are."""
    A, B = center_kernel(K1), center_kernel(K2)
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B)))


# --------------------------------------------------------------------------------------
# Synthetic exposome
# --------------------------------------------------------------------------------------

def standardize(X, ref=None):
    ref = X if ref is None else ref
    mu, sd = ref.mean(0), ref.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def make_exposome(n=200, p=10, n_active=3, rho=0.6, noise=0.5, effect="nonlinear",
                  interaction=True, seed=0):
    """Correlated exposure matrix with a sparse, optionally nonlinear response.

    Mimics the structure that makes exposome analysis hard: exposures are correlated
    within blocks (co-emitted pollutants), only a few actually matter, and the true
    exposure-response surface is nonlinear with an interaction.

    Returns (X, y, info) where info records the truth for scoring variable selection.
    """
    rng = np.random.default_rng(seed)

    # Block-correlated design: AR(1)-style correlation across columns.
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    X = rng.normal(size=(n, p)) @ L.T

    active = np.sort(rng.choice(p, size=n_active, replace=False))

    if effect == "linear":
        beta = np.zeros(p)
        beta[active] = rng.choice([-1.0, 1.0], n_active) * rng.uniform(0.8, 1.5, n_active)
        f = X @ beta
    else:
        f = np.zeros(n)
        for k, j in enumerate(active):
            if k % 3 == 0:
                f += 0.9 * np.tanh(1.5 * X[:, j])          # saturating
            elif k % 3 == 1:
                f += 0.5 * (X[:, j] ** 2 - 1.0)            # U-shaped
            else:
                f += 0.8 * np.sin(1.2 * X[:, j])           # oscillating
        beta = None

    if interaction and n_active >= 2:
        f = f + 0.6 * X[:, active[0]] * X[:, active[1]]

    f = f - f.mean()
    y = f + noise * rng.normal(size=n)

    info = {
        "active": active,
        "inactive": np.setdiff1d(np.arange(p), active),
        "f_true": f,
        "beta": beta,
        "noise": noise,
        "snr": float(f.var() / noise**2),
        "Sigma": Sigma,
    }
    return X, y, info
