"""
predict.py - refit the selected ridge config on ALL 1000 training rows and predict the test set.

    python predict.py --data_dir data --results results --out predictions
Uses results/var{1,2}_best_config.json (written by experiments.py). Writes <ROLLNO>_pred_var{1,2}.csv (single 'y' column).
"""
import argparse
import json
import os

import pandas as pd

from polyreg import PolyRegressor, load_xy, mse, r2

ROLL = "BT2024135"

def main(data_dir, results, out):
    os.makedirs(out, exist_ok=True)
    for var in (1, 2):
        path = f"{results}/var{var}_best_config.json"
        if os.path.exists(path):
            b = json.load(open(path))["best"]
        X, y = load_xy(f"{data_dir}/{ROLL}_train_var{var}.csv")
        Xt, _ = load_xy(f"{data_dir}/{ROLL}_test_var{var}.csv")
        m = PolyRegressor(b["degree"], "ridge", lam=b["lam"]).fit(X, y)
        p = m.predict(Xt)
        print(f"var{var}: degree={b['degree']} lam={b['lam']:.2e} terms={m.n_terms} | "
              f"train MSE={mse(y, m.predict(X)):.4f} R2={r2(y, m.predict(X)):.4f} | "
              f"test pred range [{p.min():.2f}, {p.max():.2f}]")
        pd.DataFrame({"y": p.numpy()}).to_csv(f"{out}/{ROLL}_pred_var{var}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="predictions")
    a = ap.parse_args()
    main(a.data_dir, a.results, a.out)