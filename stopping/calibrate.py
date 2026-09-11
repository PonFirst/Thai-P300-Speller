"""
Platt-scaling calibration for the P300 classifier score -> LLR pipeline.

Fitting the logistic regression, the reliability diagram, and ECE are
standard/off-the-shelf and implemented in full here. The actual LLR
formula that *uses* (a, b) lives in
stopping.bayesian_stopping.BayesianStoppingController.llr() and is left
for the author to derive (lecture notes Sec. 4).
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """
    Fits Platt scaling (a 1-D logistic regression) mapping raw classifier
    scores to P(target-flash | score), for use in the LLR conversion in
    BayesianStoppingController.llr().
    """

    def __init__(self, calibration_path: str):
        """
        Args:
            calibration_path: path to a CSV or pickle file containing
                calibration flashes. CSV must have columns "score" and
                "label" (1 = target flash, 0 = non-target). Pickle must
                contain a dict-like object with "scores" and "labels"
                keys, or a (scores, labels) tuple.
        """
        self.calibration_path = calibration_path
        self.model = None
        self.a = None
        self.b = None

    def load_data(self):
        """
        Load (scores, labels) from self.calibration_path.

        Returns:
            (scores: np.ndarray, labels: np.ndarray), both shape (N,).
        """
        path = self.calibration_path
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            scores = df["score"].to_numpy(dtype=float)
            labels = df["label"].to_numpy(dtype=int)
        else:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                scores = np.asarray(obj["scores"], dtype=float)
                labels = np.asarray(obj["labels"], dtype=int)
            else:
                scores, labels = obj
                scores = np.asarray(scores, dtype=float)
                labels = np.asarray(labels, dtype=int)
        return scores, labels

    def fit(self, scores: np.ndarray = None, labels: np.ndarray = None):
        """
        Fit Platt scaling: P(target | score) = sigmoid(a * score + b).

        If scores/labels are not passed explicitly, loads them via
        self.load_data().

        This is off-the-shelf sklearn logistic regression on a single
        feature -- not the novel research contribution -- so it is
        implemented in full here.

        Returns:
            (a, b): fitted slope and intercept, for use in
            BayesianStoppingController.llr().
        """
        if scores is None or labels is None:
            scores, labels = self.load_data()

        self.model = LogisticRegression()
        self.model.fit(scores.reshape(-1, 1), labels)

        self.a = float(self.model.coef_[0, 0])
        self.b = float(self.model.intercept_[0])
        return self.a, self.b

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """P(target | score) under the fitted Platt model, shape (N,)."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins.

    ECE = sum_b (n_b / N) * |acc_b - conf_b|, where for each bin b:
        acc_b  = fraction of true positives among samples in the bin
        conf_b = mean predicted probability of samples in the bin

    Args:
        probs: predicted P(target), shape (N,), values in [0, 1].
        labels: true binary labels, shape (N,).
        n_bins: number of equal-width bins over [0, 1].

    Returns:
        Scalar ECE in [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(probs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = labels[mask].mean()
        conf_b = probs[mask].mean()
        ece += (n_b / n) * abs(acc_b - conf_b)
    return float(ece)


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 10,
                              out_path: str = "results/route_a/reliability_diagram.png"):
    """
    Plot and save a reliability diagram (predicted confidence vs.
    observed accuracy per bin, with the perfect-calibration diagonal).

    Args:
        probs: predicted P(target), shape (N,).
        labels: true binary labels, shape (N,).
        n_bins: number of equal-width bins over [0, 1].
        out_path: file path to save the figure to; parent dirs are
            created if missing.

    Returns:
        out_path (str).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_acc = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append(probs[mask].mean())
        bin_acc.append(labels[mask].mean())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_centers, bin_acc, "o-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_calibration(calibration_path: str,
                     out_dir: str = "results/route_a") -> tuple:
    """
    End-to-end calibration pipeline: load data, fit Platt scaling,
    compute ECE, save a reliability diagram, and print a summary.

    Returns:
        (a, b): fitted Platt slope/intercept.
    """
    calibrator = PlattCalibrator(calibration_path)
    scores, labels = calibrator.load_data()
    a, b = calibrator.fit(scores, labels)

    probs = calibrator.predict_proba(scores)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    out_path = plot_reliability_diagram(
        probs, labels, n_bins=10,
        out_path=os.path.join(out_dir, "reliability_diagram.png"),
    )

    print(f"Platt scaling fit: a={a:.4f}, b={b:.4f}")
    print(f"Expected Calibration Error (10 bins): {ece:.4f}")
    print(f"Reliability diagram saved to: {out_path}")

    return a, b
