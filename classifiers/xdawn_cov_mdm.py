"""
xDAWN-covariance + MDM (Minimum Distance to Mean).

Pipeline: XdawnCovariances -> MDM

Conceptually different from the xDAWN+LR baseline in two ways worth being able
to explain:
  1. XdawnCovariances doesn't just spatially filter the signal -- it builds
     a spatial COVARIANCE matrix per epoch (with the training-derived
     class-mean template augmented in, per the ERP-covariance-augmentation
     trick discussed in the conversation this was scaffolded in).
  2. MDM never flattens that covariance matrix into a feature vector the
     way LR needed. It classifies directly on the Riemannian manifold of
     symmetric positive-definite (SPD) matrices, by computing the
     affine-invariant Riemannian distance from a test epoch's covariance to
     each class's geometric mean covariance, and picking the nearer one.
     No tangent-space projection happens here -- see xdawncov_tangentspace_mdmfamily.py for that.

Same GroupKFold-by-subject harness and leakage caveat as xdawn_lr.py (see that
file's docstring) -- copied here rather than imported so each day's script
stays independently runnable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for data_loader/config

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


def build_pipeline(nfilter: int = 4, metric: str = "riemann") -> Pipeline:
    """
    nfilter: number of xDAWN spatial filters per class (same meaning as
             the xDAWN+LR baseline's nfilter -- more filters = more spatial detail
             retained, at the cost of covariance matrices that are harder
             to estimate reliably from limited trials).
    metric:  the distance metric MDM uses on the SPD manifold. 'riemann'
             (affine-invariant Riemannian distance) is the standard choice
             and the one the ERP-covariance trick was designed around.
    """
    return Pipeline([
        ("cov", XdawnCovariances(nfilter=nfilter, estimator="scm", xdawn_estimator="scm")),
        ("mdm", MDM(metric=metric)),
    ])


def evaluate_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          n_splits: int = 5, **pipeline_kwargs) -> dict:
    """Same metric set and grouping logic as xdawn_lr.py, for direct comparison."""
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = build_pipeline(**pipeline_kwargs)
        pipe.fit(X[train_idx], y[train_idx])

        y_pred = pipe.predict(X[test_idx])
        classes = list(pipe.named_steps["mdm"].classes_)
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