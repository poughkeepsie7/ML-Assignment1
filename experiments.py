"""
experiments.py - model selection for one problem (var1 or var2).

    python experiments.py --var 1 --data_dir data --out results

Stage 1  80/20 split. Plain least squares for increasing degree -> train vs hold-out error.
         Also evaluates the "suggested" low-degree / feature-subset config as an ablation.
Stage 2  For every degree, tune the ridge penalty lam with 5-fold CV on the 80 % part only.
Stage 3  Pick (degree, lam) with the lowest CV MSE, refit on the 80 %, report the untouched
         20 % hold-out error, save the config for predict.py.
"""
import argparse
import json
import os

import pandas as pd
import torch

from polyreg import PolyRegressor, kfold_indices, load_xy, mse, r2

ROLL, SEED = "BT2024135", 67
LAMS = torch.logspace(-6, 1, 15).tolist()
SETTINGS = {
    1: dict(ols_degrees=range(1, 8), cv_degrees=range(2, 8),
            ablation=dict(degree=3, feature_idx=[0, 1, 2])),
    2: dict(ols_degrees=range(1, 15), cv_degrees=range(2, 15),
            ablation=dict(degree=4, feature_idx=[0])),
}


def split_80_20(n, seed=SEED):
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    k = int(0.8 * n)
    return perm[:k], perm[k:]


def main(var, data_dir, out):
    cfg = SETTINGS[var]
    os.makedirs(out, exist_ok=True)
    X, y = load_xy(f"{data_dir}/{ROLL}_train_var{var}.csv")
    dev, hold = split_80_20(len(y))
    Xd, yd, Xh, yh = X[dev], y[dev], X[hold], y[hold]
    print(f"[var{var}] dev={len(dev)} holdout={len(hold)} features={X.shape[1]}")

    # Stage 1: plain OLS degree sweep
    rows = []
    for d in cfg["ols_degrees"]:
        m = PolyRegressor(d, "ols").fit(Xd, yd)
        rows.append(dict(degree=d, n_terms=m.n_terms, train_mse=mse(yd, m.predict(Xd)),
                         holdout_mse=mse(yh, m.predict(Xh)), holdout_r2=r2(yh, m.predict(Xh))))
        print("  OLS", rows[-1])
    ols = pd.DataFrame(rows)
    ols.to_csv(f"{out}/var{var}_stage1_ols.csv", index=False)

    ab = cfg["ablation"]
    m = PolyRegressor(ab["degree"], "ols", feature_idx=ab["feature_idx"]).fit(Xd, yd)
    ablation = dict(config=ab, holdout_mse=mse(yh, m.predict(Xh)), holdout_r2=r2(yh, m.predict(Xh)))
    print("  ablation (suggested config):", ablation)

    # Stage 2: 5-fold CV over (degree, lam) on the 80 %
    folds = kfold_indices(len(yd), 5, SEED)
    rows = []
    for d in cfg["cv_degrees"]:
        for lam in LAMS:
            s = torch.tensor([mse(yd[va], PolyRegressor(d, "ridge", lam=lam).fit(Xd[tr], yd[tr]).predict(Xd[va]))
                              for tr, va in folds])
            rows.append(dict(degree=d, lam=lam, cv_mse=s.mean().item(), cv_se=(s.std() / 5 ** 0.5).item()))
        b = min((r for r in rows if r["degree"] == d), key=lambda r: r["cv_mse"])
        print(f"  CV degree {d}: best lam={b['lam']:.2e} cv_mse={b['cv_mse']:.4f}", flush=True)
    cv = pd.DataFrame(rows)
    cv.to_csv(f"{out}/var{var}_stage2_cv_grid.csv", index=False)
    per_deg = cv.loc[cv.groupby("degree").cv_mse.idxmin()].reset_index(drop=True)
    per_deg.to_csv(f"{out}/var{var}_stage2_cv_best_per_degree.csv", index=False)

    # Stage 3: choose, hold-out check
    b = cv.loc[cv.cv_mse.idxmin()]
    best = dict(degree=int(b.degree), lam=float(b.lam), cv_mse=float(b.cv_mse), cv_se=float(b.cv_se))
    m = PolyRegressor(best["degree"], "ridge", lam=best["lam"]).fit(Xd, yd)
    ph = m.predict(Xh)
    best.update(holdout_mse=mse(yh, ph), holdout_r2=r2(yh, ph), n_terms=m.n_terms)
    print("  BEST:", best)
    json.dump(dict(best=best, ablation=ablation), open(f"{out}/var{var}_best_config.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", type=int, required=True, choices=[1, 2])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    main(a.var, a.data_dir, a.out)