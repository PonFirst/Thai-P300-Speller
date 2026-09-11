"""
Day 4: xDAWN + logistic regression baseline, using pyriemann's Xdawn
implementation directly (not a from-scratch implementation -- see the
conversation this was scaffolded in for the tradeoff notes on that decision).

Pipeline: Xdawn spatial filter -> flatten -> downsample -> scale -> LR

Honest limitation worth knowing: cross-validation here groups by SUBJECT,
not by individual character-selection trial. MOABB's returned metadata
(subject/session/run) doesn't expose a per-character-trial identifier for
BNCI2014-008/009 out of the box, so flash-level leakage *within* a subject's
session isn't fully eliminated by this harness -- only cross-subject
leakage is. If you need tighter per-character grouping later, that requires
extracting event-level trial indices during epoching, which MOABB abstracts
away by default. Flagged here rather than silently overclaiming a stronger
guarantee than what's actually implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for data_loader/config

import numpy as np
from pyriemann.spatialfilters import Xdawn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Downsampler(BaseEstimator, TransformerMixin):
    """
    Downsamples the time axis of xDAWN-filtered epochs, then flattens to a
    2D feature matrix for sklearn's LogisticRegression.

    Input:  (n_epochs, n_filtered_channels, n_times)
    Output: (n_epochs, n_filtered_channels * n_times_downsampled)
    """
    def __init__(self, factor: int = 4):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ds = X[:, :, ::self.factor]  # simple decimation -- data is already
                                         # band-limited by MOABB's 1-24Hz
                                         # filter, so aliasing risk is low
        n_epochs = X_ds.shape[0]
        return X_ds.reshape(n_epochs, -1)


def build_pipeline(nfilter: int = 4, downsample_factor: int = 4, C: float = 1.0) -> Pipeline:
    """
    Full Day 4 baseline pipeline. nfilter, downsample_factor, and C (LR
    inverse regularization strength) are exposed as arguments so you can
    sweep them later rather than hardcoding choices you haven't justified.
    """
    return Pipeline([
        ("xdawn", Xdawn(nfilter=nfilter)),
        ("downsample_flatten", Downsampler(factor=downsample_factor)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=C, max_iter=2000, class_weight="balanced",
            # class_weight="balanced" matters here: your ~5:1 imbalance
            # (confirmed in your own Day 3 EDA run) means unweighted LR
            # would be biased toward always predicting NonTarget.
        )),
    ])


def evaluate_grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          n_splits: int = 5, **pipeline_kwargs) -> dict:
    """
    Runs GroupKFold cross-validation (grouped by subject -- see module
    docstring for the leakage caveat) and reports both accuracy and AUC.

    Accuracy alone is misleading at ~5:1 imbalance (see Day 3 notes) --
    balanced_accuracy_score and roc_auc_score are reported alongside it for
    exactly that reason.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        pipe = build_pipeline(**pipeline_kwargs)
        pipe.fit(X[train_idx], y[train_idx])

        y_pred = pipe.predict(X[test_idx])
        y_proba = pipe.predict_proba(X[test_idx])[:, list(pipe.classes_ if hasattr(pipe, "classes_") else pipe.named_steps["lr"].classes_).index("Target")]

        acc = (y_pred == y[test_idx]).mean()
        bal_acc = balanced_accuracy_score(y[test_idx], y_pred)
        auc = roc_auc_score((y[test_idx] == "Target").astype(int), y_proba)

        fold_results.append({"fold": fold_idx, "accuracy": acc, "balanced_accuracy": bal_acc, "auc": auc,
                               "test_subjects": sorted(set(groups[test_idx]))})
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
    results = evaluate_grouped_cv(X, y, groups, n_splits=5, nfilter=4, downsample_factor=4, C=1.0)