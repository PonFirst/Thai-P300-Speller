"""
xDAWN-covariance + tangent-space + shrinkage-regularized LDA.

Pipeline: XdawnCovariances -> TangentSpace -> LinearDiscriminantAnalysis (shrinkage)

Shares the tangent-space projection step with FgMDM (xdawn_cov_tangent.py)
-- both map each epoch's SPD covariance matrix onto the tangent plane at
the (Riemannian) mean covariance, giving a flat feature vector per epoch.
Where they differ is what happens after that projection:

  FgMDM:     learns a Fisher-discriminant rotation IN tangent space, then
             classifies by Riemannian distance back on the manifold.
  This file: treats the tangent-space vector as an ordinary Euclidean
             feature vector and fits a linear discriminant directly on it,
             with shrinkage regularization (needed because tangent-space
             dimensionality -- n_channels*(n_channels+1)/2 after xDAWN
             filtering -- can be large relative to the number of trials in
             a single GroupKFold fold).

Worth being able to explain as a genuinely different bias/variance
tradeoff, not just "another way to get a number": FgMDM's classification
rule stays intrinsically Riemannian (distance-based), while this pipeline
fully leaves the manifold after the tangent projection and lets a
regularized linear boundary do the work.

Same GroupKFold-by-subject harness as the other files -- see xdawn.py's
docstring for the leakage caveat (repeated independently here, not
imported, so this file stays runnable standalone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for data_loader/config

import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


def build_pipeline(nfilter: int = 4) -> Pipeline:
    """
    nfilter: number of xDAWN spatial filters per class (same meaning as the
             other pipelines' nfilter).

    shrinkage="auto" (Ledoit-Wolf) requires solver="lsqr" (or "eigen") --
    the default "svd" solver in sklearn doesn't support shrinkage.
    """
    return Pipeline([
        ("cov", XdawnCovariances(nfilter=nfilter, estimator="scm", xdawn_estimator="scm")),
        ("ts", TangentSpace(metric="riemann")),
        ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def evaluate_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          n_splits: int = 5, **pipeline_kwargs) -> dict:
    """Same metric set and grouping logic as the other pipeline files, for direct comparison."""
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = build_pipeline(**pipeline_kwargs)
        pipe.fit(X[train_idx], y[train_idx])

        y_pred = pipe.predict(X[test_idx])
        classes = list(pipe.named_steps["lda"].classes_)
        y_proba = pipe.predict_proba(X[test_idx])[:, classes.index("Target")]

        acc = (y_pred == y[test_idx]).mean()
        bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
        auc = roc_auc_score((y[test_idx] == "Target").astype(int), y_proba)

        fold_results.append({"fold": fold_idx, "accuracy": acc, "balanced_accuracy": bal_acc,
                               "auc": auc, "test_subjects": sorted(set(groups[test_idx]))})
        print(f"Fold {fold_idx} (test subjects={fold_results[-1]['test_subjects']}): "
              f"acc={acc:.3f}  balanced_acc={bal_acc:.3f}  AUC={auc:.3f}")

    accs = [r["accuracy"] for r in fold_results]
    bal_accs = [r["balanced_accuracy"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    print(f"\nMean +/- std across folds:")
    print(f"  accuracy:          {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  balanced accuracy: {np.mean(bal_accs):.3f} +/- {np.std(bal_accs):.3f}")
    print(f"  AUC:               {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    return {"fold_results": fold_results, "mean_accuracy": np.mean(accs),
            "mean_balanced_accuracy": np.mean(bal_accs), "mean_auc": np.mean(aucs)}


if __name__ == "__main__":
    from data_loader import load_arrays

    X, y, metadata = load_arrays("BNCI2014_008", subjects=[1, 2, 3, 4, 5])
    groups = metadata["subject"].values

    print(f"X: {X.shape}, groups (subjects): {sorted(set(groups))}\n")
    results = evaluate_grouped_cv(X, y, groups, n_splits=5, nfilter=4)
