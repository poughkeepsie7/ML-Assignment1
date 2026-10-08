"""
polyreg.py - Polynomial regression in PyTorch (OLS and ridge only).

* poly_exponents / poly_features : multivariate polynomial expansion (total degree <= d)
* PolyRegressor                  : standardise -> expand -> linear fit
      solver="ols"   : least squares via torch.linalg.lstsq (min-norm if under-determined)
      solver="ridge" : closed form  w = (P^T P / n + lam*I)^-1 P^T y / n
* mse / r2, load_xy, kfold_indices : small helpers
"""
from itertools import combinations_with_replacement

import pandas as pd
import torch

DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def load_xy(path):
    """Read a csv; returns (X, y) with y=None if there is no 'y' column."""
    df = pd.read_csv(path)
    y = torch.tensor(df["y"].values, dtype=DTYPE) if "y" in df.columns else None
    X = torch.tensor(df.drop(columns="y", errors="ignore").values, dtype=DTYPE)
    return X, y


def mse(y, p):
    return torch.mean((y - p) ** 2).item()


def r2(y, p):
    return (1 - torch.sum((y - p) ** 2) / torch.sum((y - y.mean()) ** 2)).item()


def kfold_indices(n, k, seed):
    """List of (train_idx, val_idx) for shuffled k-fold CV."""
    g = torch.Generator().manual_seed(seed)
    folds = torch.tensor_split(torch.randperm(n, generator=g), k)
    return [(torch.cat([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]


def poly_exponents(n_features, degree):
    """All exponent vectors e with 1 <= sum(e) <= degree (no constant term)."""
    exps = []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            e = [0] * n_features
            for i in combo:
                e[i] += 1
            exps.append(e)
    return torch.tensor(exps, dtype=torch.long)


def poly_features(X, exps):
    """X: (n, p); exps: (m, p) -> (n, m) matrix of monomials."""
    pows = [torch.ones_like(X)]
    for _ in range(int(exps.max())):
        pows.append(pows[-1] * X)
    pows = torch.stack(pows)                       # (max_pow+1, n, p)
    cols = torch.ones(X.shape[0], exps.shape[0])
    for j in range(X.shape[1]):
        cols = cols * pows[exps[:, j], :, j].T
    return cols


class PolyRegressor:
    def __init__(self, degree, solver="ols", lam=0.0, feature_idx=None):
        self.degree, self.solver, self.lam, self.feature_idx = degree, solver, lam, feature_idx

    def _design(self, X, fit=False):
        if self.feature_idx is not None:
            X = X[:, self.feature_idx]
        if fit:
            self.exps = poly_exponents(X.shape[1], self.degree)
        P = poly_features(X, self.exps)
        if fit:                                     # statistics from training data only
            self.mu, self.sd = P.mean(0), P.std(0).clamp_min(1e-12)
        return (P - self.mu) / self.sd

    def fit(self, X, y):
        P = self._design(X, fit=True)
        self.y_mean = y.mean()
        yc = y - self.y_mean
        n, m = P.shape
        if self.solver == "ols":
            self.w = torch.linalg.lstsq(P, yc.unsqueeze(1), driver="gelsd").solution.squeeze(1)
        elif self.solver == "ridge":
            self.w = torch.linalg.solve(P.T @ P / n + self.lam * torch.eye(m), P.T @ yc / n)
        else:
            raise ValueError(self.solver)
        return self

    def predict(self, X):
        return self._design(X) @ self.w + self.y_mean

    @property
    def n_terms(self):
        return self.w.numel()