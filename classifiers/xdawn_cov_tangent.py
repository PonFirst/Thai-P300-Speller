"""
xDAWN-covariance + tangent-space-aware MDM-family classifiers.

Two classifiers here, both building on plain MDM (xdawncov_mdm.py) by incorporating
tangent-space geometry, but in different ways -- worth being able to
distinguish them, not just run both and compare numbers:

  FgMDM (Fisher-geodesic MDM):
    Projects covariance matrices onto the tangent plane, applies a
    Fisher-discriminant-style linear transform THERE (using class labels,
    unlike plain MDM which only uses labels to compute the class means),
    then maps back and classifies by distance to the transformed class
    means. Effectively: MDM, but with a learned tangent-space rotation that
    pushes classes further apart before measuring distance.

  MeanField:
    Instead of one geometric (Riemannian) mean per class, computes several
    "power means" per class (controlled by power_list, default [-1, 0, 1] --
    0 is the standard Riemannian/geometric mean; -1 and 1 are harmonic- and
    arithmetic-like generalizations that weight outlier trials differently)
    and combines them. This is more robust when a class's trials don't
    cluster tightly around one single mean -- plausible for Target epochs
    given the single-trial noise you already characterized in your Day 3
    SNR curve. Matches the "MDMF"/"MF" pipelines from the benchmark
    comparison a few messages back.

Same GroupKFold-by-subject harness as the other files -- see xdawn_lr.py's docstring for
the leakage caveat (repeated independently here, not imported, so this
file stays runnable standalone).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for data_loader/config

import numpy as np
from pyriemann.classification import FgMDM, MeanField
from pyriemann.estimation import XdawnCovariances
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

CLASSIFIERS = {
    "FgMDM": lambda: FgMDM(metric="riemann"),
    "MeanField": lambda: MeanField(power_list=[-1, 0, 1], metric="riemann"),
}


def build_pipeline(classifier_name: str, nfilter: int = 4) -> Pipeline:
    if classifier_name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier {classifier_name!r}. Choose from {list(CLASSIFIERS)}")
    return Pipeline([
        ("cov", XdawnCovariances(nfilter=nfilter, estimator="scm", xdawn_estimator="scm")),
        ("clf", CLASSIFIERS[classifier_name]()),
    ])


def evaluate_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          classifier_name: str, n_splits: int = 5, nfilter: int = 4) -> dict:
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = build_pipeline(classifier_name, nfilter=nfilter)
        pipe.fit(X[train_idx], y[train_idx])

        y_pred = pipe.predict(X[test_idx])
        classes = list(pipe.named_steps["clf"].classes_)
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

    return {"fold_results": fold_results, "mean_accuracy": np.mean(accs),
            "mean_balanced_accuracy": np.mean(bal_accs), "mean_auc": np.mean(aucs)}


def run_all(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, nfilter: int = 4) -> dict:
    """Runs both FgMDM and MeanField back to back for direct comparison."""
    all_results = {}
    for name in CLASSIFIERS:
        print(f"\n--- {name} ---")
        all_results[name] = evaluate_grouped_cv(X, y, groups, classifier_name=name,
                                                   n_splits=n_splits, nfilter=nfilter)

    print(f"\n--- Summary ---")
    for name, res in all_results.items():
        print(f"  {name:12s}  AUC={res['mean_auc']:.3f}  balanced_acc={res['mean_balanced_accuracy']:.3f}")

    return all_results


if __name__ == "__main__":
    from data_loader import load_arrays

    X, y, metadata = load_arrays("BNCI2014_008", subjects=[1, 2, 3, 4, 5])
    groups = metadata["subject"].values

    print(f"X: {X.shape}, groups (subjects): {sorted(set(groups))}")
    results = run_all(X, y, groups, n_splits=5, nfilter=4)