"""
Runs all 5 classifier pipelines on IDENTICAL GroupKFold-by-subject CV folds,
for a fair head-to-head comparison within one dataset.

get_pipelines() is the single source of truth for "what are the 5
pipelines" -- other scripts (e.g. all_classifiers.py) should import and
reuse it rather than redefining pipelines, to avoid silent drift between
files. Each entry wraps the build_pipeline() already defined and validated
in that pipeline's own file:

  xDAWN+LR             -> xdawn.py
  xDAWNcov+MDM          -> xdawn_cov_mdm.py
  xDAWNcov+FgMDM        -> xdawn_cov_tangent.py ("FgMDM")
  xDAWNcov+MeanField    -> xdawn_cov_tangent.py ("MeanField")
  xDAWNcov+TS+RegLDA    -> xdawn_cov_ts_regLDA.py

Same GroupKFold-by-subject leakage caveat as every individual pipeline file
-- see xdawn.py's docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for data_loader/config

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import xdawn
import xdawn_cov_mdm
import xdawn_cov_tangent
import xdawn_cov_ts_regLDA


def get_pipelines(nfilter: int = 4) -> dict:
    """
    Returns {pipeline_name: zero-arg callable that builds a fresh, unfit
    sklearn Pipeline}. Callables (not pre-built instances) so run_benchmark
    can build a clean estimator per fold without cross-fold state leaking
    through a shared fitted object.
    """
    return {
        "xDAWN+LR": lambda: xdawn.build_pipeline(nfilter=nfilter, downsample_factor=4, C=1.0),
        "xDAWNcov+MDM": lambda: xdawn_cov_mdm.build_pipeline(nfilter=nfilter),
        "xDAWNcov+FgMDM": lambda: xdawn_cov_tangent.build_pipeline("FgMDM", nfilter=nfilter),
        "xDAWNcov+MeanField": lambda: xdawn_cov_tangent.build_pipeline("MeanField", nfilter=nfilter),
        "xDAWNcov+TS+RegLDA": lambda: xdawn_cov_ts_regLDA.build_pipeline(nfilter=nfilter),
    }


def _last_step_classes(pipe) -> list:
    """The fitted final estimator's classes_, regardless of its step name."""
    return list(pipe.steps[-1][1].classes_)


def run_benchmark(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_splits: int = 5, nfilter: int = 4) -> dict:
    """
    Fixes GroupKFold folds ONCE up front so every pipeline trains/tests on
    identical folds -- required for a fair comparison, not just convenient.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups=groups))

    pipelines = get_pipelines(nfilter=nfilter)
    all_results = {}

    for name, build_fn in pipelines.items():
        print(f"\n--- {name} ---")
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            pipe = build_fn()
            pipe.fit(X[train_idx], y[train_idx])

            y_pred = pipe.predict(X[test_idx])
            classes = _last_step_classes(pipe)
            y_proba = pipe.predict_proba(X[test_idx])[:, classes.index("Target")]

            acc = (y_pred == y[test_idx]).mean()
            bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
            auc = roc_auc_score((y[test_idx] == "Target").astype(int), y_proba)

            fold_results.append({"fold": fold_idx, "accuracy": acc, "balanced_accuracy": bal_acc,
                                   "auc": auc, "test_subjects": sorted(set(groups[test_idx]))})
            print(f"  Fold {fold_idx} (test subjects={fold_results[-1]['test_subjects']}): "
                  f"acc={acc:.3f}  balanced_acc={bal_acc:.3f}  AUC={auc:.3f}")

        accs = [r["accuracy"] for r in fold_results]
        bal_accs = [r["balanced_accuracy"] for r in fold_results]
        aucs = [r["auc"] for r in fold_results]
        print(f"  Mean +/- std -- accuracy: {np.mean(accs):.3f}+/-{np.std(accs):.3f}  "
              f"balanced_acc: {np.mean(bal_accs):.3f}+/-{np.std(bal_accs):.3f}  "
              f"AUC: {np.mean(aucs):.3f}+/-{np.std(aucs):.3f}")

        all_results[name] = {"fold_results": fold_results, "mean_accuracy": np.mean(accs),
                               "mean_balanced_accuracy": np.mean(bal_accs), "mean_auc": np.mean(aucs)}

    print(f"\n--- Summary ---")
    df = pd.DataFrame([
        {"Pipeline": name, "AUC": res["mean_auc"], "BalAcc": res["mean_balanced_accuracy"],
         "Accuracy": res["mean_accuracy"]}
        for name, res in all_results.items()
    ]).sort_values("AUC", ascending=False)
    print(df.to_string(index=False, float_format="{:.3f}".format))

    return all_results


if __name__ == "__main__":
    import argparse

    from data_loader import load_arrays

    parser = argparse.ArgumentParser(description="Benchmark all 5 classifier pipelines on one dataset.")
    parser.add_argument("--dataset", default="BNCI2014_008", choices=["BNCI2014_008", "BNCI2014_009"])
    parser.add_argument("--subjects", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--nfilter", type=int, default=4)
    args = parser.parse_args()

    X, y, metadata = load_arrays(args.dataset, subjects=args.subjects)
    groups = metadata["subject"].values

    print(f"X: {X.shape}, groups (subjects): {sorted(set(groups))}")
    results = run_benchmark(X, y, groups, n_splits=args.n_splits, nfilter=args.nfilter)
