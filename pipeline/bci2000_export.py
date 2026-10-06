"""
Export a fitted classifier to a BCI2000 parameter file for the RiemannP300Classifier filter
(bci2000/CheckerboardSignalProcessing), and the reference arithmetic that filter must reproduce.

    python -m pipeline.bci2000_export --checkpoint models/<id>_classifier.joblib \\
        --calibration models/<id>_platt.json --out models/<id>_bci2000.prm

fit_from_recording writes this file itself when it fits from a BCI2000 .dat recording.

What is exported (everything the online filter needs, nothing it would have to re-derive):
  - the epoch: EpochSpec's channels, sampling rate, n_times, n_pad, optional baseline;
  - the band-pass: the SOS coefficients, sosfilt_zi initial conditions and the odd-extension
    length that scipy.signal.sosfiltfilt uses (computed here by scipy, not redesigned in C++);
  - XdawnCovariances: the spatial filters (Xd_.filters_) and prototypes (P_);
  - TangentSpace: reference_^(-1/2), computed with pyriemann's own invsqrtm;
  - the LDA weights and intercept; and Platt a, b for the stopping rule.

reference_score() is the specification of the C++ arithmetic, step by step in numpy, and is
tested against the pipeline's own decision_function. The C++ is tested against both.

Only the exact pipeline of classifiers/xdawn_cov_ts_regLDA.build_pipeline() is accepted: any other
step, estimator or setting is refused rather than approximated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

from pipeline.classifier_io import load_checkpoint
from pipeline.constants import RAILED_PTP_UV, SANITY_PTP_MIN_UV
from pipeline.epoching import EpochSpec, _bandpass_sos, _sosfiltfilt_padlen, is_rank_deficient
from pipeline.manifest import load_calibration_file, sha256_file
from pipeline.quality import FLAT_SEGMENT_S, GATE_POLICIES, GATE_SKIP, MIN_SAMPLES_FOR_QUALITY, gate_epoch

PRM_SECTION = "Filtering:Riemann%20P300%20Classifier"
EXPORT_FORMAT = "p300-bci2000-riemann-v1"


class ExportError(ValueError):
    """The checkpoint is not the pipeline the BCI2000 filter implements."""


@dataclass
class ExportedModel:
    channels: tuple
    sfreq: float
    n_times: int
    n_pad: int
    baseline: tuple | None        # (i0, i1) sample indices within the epoch, or None
    sos: np.ndarray               # (n_sections, 6); empty when the EpochSpec has no band
    zi: np.ndarray                # (n_sections, 2)
    padlen: int                   # sosfiltfilt's odd-extension length
    xdawn_filters: np.ndarray     # (n_filters, n_channels)
    prototypes: np.ndarray        # (n_proto, n_times)
    ref_invsqrt: np.ndarray       # (n, n), n = n_proto + n_filters
    lda_weights: np.ndarray       # (n * (n + 1) / 2,)
    lda_intercept: float
    platt_a: float
    platt_b: float
    model_id: str                 # sha256 of the checkpoint file
    gate_policy: str = GATE_SKIP  # pipeline/quality.py: "skip" rejects a flash with a red channel
    gate_flat_uv: float = SANITY_PTP_MIN_UV
    gate_railed_uv: float = RAILED_PTP_UV
    gate_segment: int = 25        # samples per flat-segment check (FLAT_SEGMENT_S at sfreq)


def _pyriemann_invsqrtm(m: np.ndarray) -> np.ndarray:
    try:
        from pyriemann.geometry.base import invsqrtm
    except ImportError:              # pyriemann < 0.12 layout
        from pyriemann.utils.base import invsqrtm
    return invsqrtm(m)


def extract_model(checkpoint_path, calibration_path, gate_policy: str = GATE_SKIP) -> ExportedModel:
    ckpt = load_checkpoint(checkpoint_path)
    cal = load_calibration_file(calibration_path)
    model_id = sha256_file(checkpoint_path)
    if cal.get("classifier_sha256") != model_id:
        raise ExportError(f"the calibration {Path(calibration_path).name} was fitted for another checkpoint "
                          f"({str(cal.get('classifier_sha256'))[:12]}..., this one is {model_id[:12]}...)")
    if gate_policy not in GATE_POLICIES:
        raise ExportError(f"gate policy {gate_policy!r} is not one of {GATE_POLICIES}")
    spec: EpochSpec = ckpt.epoch_spec
    if spec.tmin != 0:
        raise ExportError(f"EpochSpec.tmin is {spec.tmin}; the BCI2000 filter cuts every epoch from the flash onset")
    pipe = ckpt.pipeline
    names = [n for n, _ in getattr(pipe, "steps", [])]
    if names != ["cov", "ts", "lda"]:
        raise ExportError(f"pipeline steps are {names}; the BCI2000 filter implements exactly cov -> ts -> lda")
    cov, ts, lda = (pipe.named_steps[n] for n in names)

    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    if type(cov) is not XdawnCovariances or type(ts) is not TangentSpace or type(lda) is not LinearDiscriminantAnalysis:
        raise ExportError(f"unexpected step types {[type(s).__name__ for s in (cov, ts, lda)]}")
    cp = cov.get_params()
    if cp.get("estimator") != "scm" or not cp.get("applyfilters", True) or cp.get("baseline_cov") is not None:
        raise ExportError(f"XdawnCovariances settings {cp} are not the ones the filter implements "
                          "(estimator='scm', applyfilters=True, baseline_cov=None)")
    if getattr(cov, "kwds", {}):
        raise ExportError(f"XdawnCovariances has estimator options {cov.kwds}; the filter implements plain scm")
    tp = ts.get_params()
    if tp.get("metric") != "riemann" or tp.get("tsupdate"):
        raise ExportError(f"TangentSpace settings {tp} are not the ones the filter implements (riemann, no tsupdate)")
    if list(lda.classes_) != [0, 1] or lda.coef_.shape[0] != 1:
        raise ExportError(f"LDA must be binary with classes [0, 1] (1 = target), got {list(lda.classes_)}")

    filters = np.asarray(cov.Xd_.filters_, dtype=np.float64)
    protos = np.asarray(cov.P_, dtype=np.float64)
    ref = np.asarray(ts.reference_, dtype=np.float64)
    n = protos.shape[0] + filters.shape[0]
    if filters.shape[1] != len(spec.channels) or protos.shape[1] != spec.n_times or ref.shape != (n, n):
        raise ExportError(f"shapes do not fit the EpochSpec: filters {filters.shape}, prototypes {protos.shape}, "
                          f"reference {ref.shape}, {len(spec.channels)} channels x {spec.n_times} samples")
    weights = np.asarray(lda.coef_[0], dtype=np.float64)
    if weights.shape != (n * (n + 1) // 2,):
        raise ExportError(f"LDA has {weights.size} weights; a {n}x{n} tangent space has {n * (n + 1) // 2}")

    if spec.band is not None:
        sos = np.asarray(_bandpass_sos(spec.filter_order, spec.band[0], spec.band[1], spec.sfreq), dtype=np.float64)
        zi = np.asarray(sosfilt_zi(sos), dtype=np.float64)
        padlen = int(_sosfiltfilt_padlen(sos))
        if not np.allclose(sos[:, 3], 1.0):
            raise ExportError("SOS sections must have a0 = 1")
    else:
        sos, zi, padlen = np.zeros((0, 6)), np.zeros((0, 2)), 0
    baseline = spec._baseline_indices() if spec.baseline is not None else None

    return ExportedModel(
        channels=tuple(spec.channels), sfreq=float(spec.sfreq), n_times=spec.n_times, n_pad=spec.n_pad,
        baseline=baseline, sos=sos, zi=zi, padlen=padlen, xdawn_filters=filters, prototypes=protos,
        ref_invsqrt=np.asarray(_pyriemann_invsqrtm(ref), dtype=np.float64), lda_weights=weights,
        lda_intercept=float(lda.intercept_[0]), platt_a=float(cal["a"]), platt_b=float(cal["b"]),
        model_id=model_id, gate_policy=gate_policy, gate_flat_uv=float(SANITY_PTP_MIN_UV),
        gate_railed_uv=float(RAILED_PTP_UV),
        gate_segment=max(MIN_SAMPLES_FOR_QUALITY, int(round(FLAT_SEGMENT_S * spec.sfreq))))


# ---------------------------------------------------------------------------
# the arithmetic, step by step: the specification of the C++ filter
# ---------------------------------------------------------------------------

def zero_phase_filter(x: np.ndarray, m: ExportedModel) -> np.ndarray:
    """scipy.signal.sosfiltfilt(m.sos, x, axis=-1) written out: odd extension, forward pass from
    the steady state of the first sample, backward pass from the steady state of the last."""
    if m.sos.shape[0] == 0:
        return x.copy()
    n = m.padlen
    left = 2 * x[:, :1] - x[:, n:0:-1]
    right = 2 * x[:, -1:] - x[:, -2:-(n + 2):-1]
    ext = np.concatenate([left, x, right], axis=1)
    y = np.empty_like(ext)
    for c in range(ext.shape[0]):
        f, _ = sosfilt(m.sos, ext[c], zi=m.zi * ext[c, 0])
        b, _ = sosfilt(m.sos, f[::-1], zi=m.zi * f[-1])
        y[c] = b[::-1]
    return y[:, n:-n]


def upper_weighted(s: np.ndarray) -> np.ndarray:
    """pyriemann upper(): row-major upper triangle, off-diagonal entries times sqrt(2)."""
    i, j = np.triu_indices(s.shape[0])
    return s[i, j] * np.where(i == j, 1.0, math.sqrt(2.0))


def reference_score(m: ExportedModel, segment: np.ndarray) -> float:
    """
    LDA decision score for one padded raw segment shaped (n_pad + n_times + n_pad, n_channels) in uV,
    exactly as the BCI2000 filter computes it.
    """
    x = np.asarray(segment, dtype=np.float64).T                        # (C, n_padded)
    x = zero_phase_filter(x, m)[:, m.n_pad:m.n_pad + m.n_times]         # band-pass, crop
    if m.baseline is not None:
        i0, i1 = m.baseline
        x = x - x[:, i0:i1].mean(axis=1, keepdims=True)
    z = np.concatenate([m.prototypes, m.xdawn_filters @ x], axis=0)     # prototypes first (covariances_EP)
    z = z - z.mean(axis=1, keepdims=True)
    cov = (z @ z.T) / z.shape[1]                                        # scm, 1/n_times
    s = m.ref_invsqrt @ cov @ m.ref_invsqrt
    s = 0.5 * (s + s.T)
    w, v = np.linalg.eigh(s)
    log_s = (v * np.log(w)) @ v.T
    return float(upper_weighted(log_s) @ m.lda_weights + m.lda_intercept)


def preprocessed_epoch(m: ExportedModel, segment: np.ndarray) -> np.ndarray:
    """Band-passed, cropped, baseline-corrected epoch (n_channels, n_times), as epoching.preprocess_segment."""
    x = zero_phase_filter(np.asarray(segment, dtype=np.float64).T, m)[:, m.n_pad:m.n_pad + m.n_times]
    if m.baseline is not None:
        i0, i1 = m.baseline
        x = x - x[:, i0:i1].mean(axis=1, keepdims=True)
    return x


def reference_decision(m: ExportedModel, segment: np.ndarray) -> tuple[float | None, str | None]:
    """
    What the BCI2000 filter decides for one padded raw window, in its order: (score, None), or
    (None, reason) with reason one of "non_finite", "channel_quality", "rank_deficient" -- the gate
    before the rank check, as in the Python service.
    The quality gate is pipeline.quality.gate_epoch itself, so the C++ port is tested against the
    exact rule the Python service applies.
    """
    seg = np.asarray(segment, dtype=np.float64)
    if not np.all(np.isfinite(seg)):
        return None, "non_finite"
    x = preprocessed_epoch(m, seg)
    if m.gate_policy == GATE_SKIP and not gate_epoch(x, list(m.channels), m.gate_policy, m.sfreq).accept:
        return None, "channel_quality"
    if is_rank_deficient(x):
        return None, "rank_deficient"
    return reference_score(m, seg), None


# ---------------------------------------------------------------------------
# the parameter file
# ---------------------------------------------------------------------------

def _num(x: float) -> str:
    v = float(x)
    if not math.isfinite(v):
        raise ExportError(f"non-finite value {v} in the model")
    return repr(v)


def _matrix(name: str, a: np.ndarray, comment: str) -> str:
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    rows, cols = (0, 0) if a.size == 0 else a.shape
    values = " ".join(_num(v) for v in a.ravel()) if a.size else ""
    return f"{PRM_SECTION} matrix {name}= {rows} {cols} {values} // {comment}".replace("  //", " //")


def _list(name: str, ptype: str, values, comment: str) -> str:
    vals = list(values)
    return f"{PRM_SECTION} {ptype} {name}= {len(vals)} {' '.join(vals)} // {comment}".replace("  //", " //")


def _bci_encode(value: str) -> str:
    """BCI2000 EncodedString: printable ASCII passes except space, '%' and the brackets BCI2000 reads
    as list/matrix delimiters even inside a string value ({}[]: an unescaped '{' swallows the
    following lines of a parameter file); every other byte is %XX."""
    if value == "":
        return "%"
    if "\x00" in value:
        raise ExportError("a NUL character cannot be stored in a BCI2000 parameter (its decoder drops it)")
    return "".join(c if (0x21 <= ord(c) <= 0x7E and c not in "%{}[]") else "".join(f"%{b:02X}" for b in c.encode())
                   for c in value)


def render_prm(m: ExportedModel, provenance: dict) -> str:
    base = "none" if m.baseline is None else f"{m.baseline[0]} {m.baseline[1]}"
    lines = [
        f"{PRM_SECTION} string RiemannFormat= {EXPORT_FORMAT} // export format (readonly)",
        f"{PRM_SECTION} string RiemannModelID= {m.model_id} // sha256 of the fitted checkpoint",
        f"{PRM_SECTION} string RiemannProvenance= {_bci_encode(json.dumps(provenance, sort_keys=True))} "
        "// what the model was fitted on (JSON)",
        _list("RiemannChannels", "list", [_bci_encode(c) for c in m.channels],
              "channel names, in order; must equal the source's ChannelNames"),
        f"{PRM_SECTION} float RiemannSamplingRate= {_num(m.sfreq)} // Hz; must equal SamplingRate",
        f"{PRM_SECTION} int RiemannEpochSamples= {m.n_times} // samples from the flash onset",
        f"{PRM_SECTION} int RiemannPadSamples= {m.n_pad} // raw samples filtered on each side of the epoch",
        _list("RiemannBaseline", "intlist", [] if m.baseline is None else [str(m.baseline[0]), str(m.baseline[1])],
              f"baseline sample range within the epoch, empty for none ({base})"),
        _matrix("RiemannFilterSOS", m.sos, "band-pass second-order sections b0 b1 b2 a0 a1 a2 (scipy)"),
        _matrix("RiemannFilterZI", m.zi, "sosfilt_zi steady-state initial conditions (scipy)"),
        f"{PRM_SECTION} int RiemannFilterPadlen= {m.padlen} // sosfiltfilt odd-extension length",
        _matrix("RiemannXdawnFilters", m.xdawn_filters, "xDAWN spatial filters, one per row"),
        _matrix("RiemannPrototypes", m.prototypes, "xDAWN prototype responses, one per row"),
        _matrix("RiemannRefInvSqrt", m.ref_invsqrt, "tangent-space reference to the power -1/2"),
        _list("RiemannLDAWeights", "floatlist", [_num(v) for v in m.lda_weights],
              "LDA weights on the weighted upper triangle"),
        f"{PRM_SECTION} float RiemannLDAIntercept= {_num(m.lda_intercept)} // LDA intercept",
        f"{PRM_SECTION} float RiemannPlattA= {_num(m.platt_a)} // P(target) = 1 / (1 + exp(-(a*score + b)))",
        f"{PRM_SECTION} float RiemannPlattB= {_num(m.platt_b)} // Platt intercept",
        f"{PRM_SECTION} int RiemannQualityGate= {1 if m.gate_policy == GATE_SKIP else 0} 1 0 1 "
        "// reject a flash with a flat or railed channel (pipeline/quality.py gate_epoch, skip policy) (boolean)",
        f"{PRM_SECTION} float RiemannGateFlatUV= {_num(m.gate_flat_uv)} // a channel or segment below this "
        "peak-to-peak (uV) is flat",
        f"{PRM_SECTION} float RiemannGateRailedUV= {_num(m.gate_railed_uv)} // a channel above this peak-to-peak "
        "(uV) is railed",
        f"{PRM_SECTION} int RiemannGateSegmentSamples= {m.gate_segment} // flat-segment length, counted back "
        "from the epoch's last sample",
    ]
    return "\n".join(lines) + "\n"


def stopping_lines(stopping: dict, lm_file: Path | None) -> list[str]:
    """CheckerboardSpeller's Bayesian-stopping and language-model parameters (bci2000/CheckerboardSpeller):
    the manifest's stopping settings, and the Transformer export (or the uniform prior when None)."""
    sec, lmsec = "Application:Bayesian%20Stopping", "Application:Language%20Model"
    return [
        f"{sec} int StoppingRule= 1 1 0 1 // Bayesian dynamic stopping (enumeration)",
        f"{sec} float StoppingLambda= {_num(stopping['lam'])} 0.7 0 1 // prior blend weight",
        f"{sec} float StoppingTheta= {_num(stopping['theta'])} 0.9 0 1 // stopping threshold",
        f"{sec} int StoppingTMin= {int(stopping['t_min'])} 6 0 % // flashes before stopping is allowed",
        f"{sec} int StoppingTMax= {int(stopping['t_max'])} 120 1 % // forced stop",
        f"{sec} float StoppingLLRClip= {_num(stopping['llr_clip'])} 8 0 % // per-flash LLR clip",
        f"{lmsec} int LanguageModel= {1 if lm_file is not None else 0} 1 0 1 "
        "// 0 uniform, 1 Thai Transformer (enumeration)",
        f"{lmsec} string LanguageModelFile= {_bci_encode(str(Path(lm_file).resolve())) if lm_file else '%'} "
        "% % % // .lmbin written by pipeline/bci2000_lm_export.py",
    ]


def export_prm(checkpoint_path, calibration_path, out_path, provenance: dict | None = None,
               gate_policy: str = GATE_SKIP, extra_lines: list[str] | None = None) -> Path:
    m = extract_model(checkpoint_path, calibration_path, gate_policy)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    text = render_prm(m, provenance or {}) + "".join(line + "\n" for line in (extra_lines or []))
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export a fitted classifier to a BCI2000 parameter file.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    try:
        out = export_prm(args.checkpoint, args.calibration, args.out)
    except (ExportError, ValueError, OSError) as exc:
        print(f"bci2000_export: error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
