"""
session_manifest.json: the model artifacts and settings a run uses. The
recorder, inference service, and GUI all load it; none hardcode a model path.

Schema version 1:

    {
      "schema_version": 1,
      "characters":  {"path": "collection/characters_th.json", "sha256": "<64 hex>"},
      "classifier":  {"path": "models/subject_001_classifier.joblib", "sha256": "<64 hex>"},
      "calibration": {"path": "models/subject_001_platt.json", "sha256": "<64 hex>"},
      "language_model": {"type": "uniform"},
      "stopping": {"lam": 0.7, "theta": 0.9, "t_min": 6, "t_max": 170, "llr_clip": 8.0},
      "quality_gate": {"policy": "skip"}
    }

language_model is {"type": "uniform"}, or
    {"type": "ngram", "corpus": {"path": ..., "sha256": ...},
     "order": 4, "alpha": 0.1, "backoff_weight": 0.4}, or
    {"type": "transformer", "checkpoint": {"path": ..., "sha256": ...}}
"uniform" is a declared choice, not a fallback. "transformer" loads a
stopping/gpt_dev.ipynb checkpoint via stopping.transformer_lm.ThaiTransformerLM
(pipeline/language_model.py's _build_transformer) -- its architecture
hyperparameters travel inside the checkpoint itself, not duplicated here.

Enforced at load time:
  - relative paths resolve against the manifest's directory, not the cwd;
  - every artifact's sha256 is re-checked, so a checkpoint replaced behind an
    unchanged path is an error rather than a silent swap;
  - the calibration file records the sha256 of the classifier it was fitted
    on, which must match the manifest's classifier;
  - the characters file holds exactly 72 unique entries.

The fingerprint is the sha256 of the canonical JSON. The manifest embeds every
artifact hash, so two processes reporting the same fingerprint are using
byte-identical models and settings.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.constants import N_CHARS
from pipeline.quality import GATE_POLICIES, GATE_SKIP

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "session_manifest.json"

LM_TYPES = ("uniform", "ngram", "transformer")
CALIBRATION_FORMAT = "p300-platt-v1"
SCORE_METHOD = "decision_function"

DEFAULT_STOPPING = {"lam": 0.7, "theta": 0.90, "t_min": 6, "t_max": 170, "llr_clip": 8.0}
DEFAULT_QUALITY_GATE = {"policy": GATE_SKIP}
DEFAULT_NGRAM = {"order": 4, "alpha": 0.1, "backoff_weight": 0.4}

_TOP_LEVEL_KEYS = {"schema_version", "characters", "classifier", "calibration",
                   "language_model", "stopping", "quality_gate"}
_CALIBRATION_REQUIRED = {"format", "a", "b", "score_method", "classifier_sha256"}
_CALIBRATION_OPTIONAL = {"ece", "n_scores", "n_targets", "created_utc", "source_recording_sha256"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(Exception):
    """The session manifest, or an artifact it names, is invalid."""


class ManifestMismatchError(ManifestError):
    """Two processes are running with different session manifests."""


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint_of(data: dict) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def short_fingerprint(fingerprint: str | None) -> str:
    return fingerprint[:12] if fingerprint else "<none>"


def _fail(where: str, message: str):
    raise ManifestError(f"{where}: {message}")


def _exact_keys(obj, required: set, where: str, optional: set = frozenset()):
    if not isinstance(obj, dict):
        _fail(where, f"must be an object, got {type(obj).__name__}")
    missing = required - obj.keys()
    extra = obj.keys() - required - optional
    if missing:
        _fail(where, f"missing key(s) {sorted(missing)}")
    if extra:
        _fail(where, f"unexpected key(s) {sorted(extra)}")


def _number(value, where, *, lo=None, hi=None, lo_open=False, hi_open=False, integer=False):
    kind = "an integer" if integer else "a number"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, f"must be {kind}, got {value!r}")
    if integer and not isinstance(value, int):
        _fail(where, f"must be {kind}, got {value!r}")
    if not math.isfinite(value):
        _fail(where, f"must be finite, got {value!r}")
    if lo is not None and (value <= lo if lo_open else value < lo):
        _fail(where, f"must be {'>' if lo_open else '>='} {lo}, got {value!r}")
    if hi is not None and (value >= hi if hi_open else value > hi):
        _fail(where, f"must be {'<' if hi_open else '<='} {hi}, got {value!r}")


def _sha256_field(value, where):
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        _fail(where, "must be a 64-character lowercase hex sha256")


def _artifact(entry, where):
    _exact_keys(entry, {"path", "sha256"}, where)
    if not isinstance(entry["path"], str) or not entry["path"]:
        _fail(f"{where}.path", "must be a non-empty string")
    _sha256_field(entry["sha256"], f"{where}.sha256")


def validate_manifest_dict(data: Any) -> None:
    """Structure, types, and ranges only -- no file access."""
    _exact_keys(data, _TOP_LEVEL_KEYS, "manifest")
    version = data["schema_version"]
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        _fail("manifest.schema_version", f"must be {SCHEMA_VERSION}, got {version!r}")
    for key in ("characters", "classifier", "calibration"):
        _artifact(data[key], f"manifest.{key}")

    lm = data["language_model"]
    if not isinstance(lm, dict):
        _fail("manifest.language_model", f"must be an object, got {type(lm).__name__}")
    if lm.get("type") not in LM_TYPES:
        _fail("manifest.language_model.type", f"must be one of {list(LM_TYPES)}, got {lm.get('type')!r}")
    if lm["type"] == "uniform":
        _exact_keys(lm, {"type"}, "manifest.language_model")
    elif lm["type"] == "ngram":
        _exact_keys(lm, {"type", "corpus", *DEFAULT_NGRAM}, "manifest.language_model")
        _artifact(lm["corpus"], "manifest.language_model.corpus")
        _number(lm["order"], "manifest.language_model.order", lo=1, integer=True)
        _number(lm["alpha"], "manifest.language_model.alpha", lo=0, lo_open=True)
        _number(lm["backoff_weight"], "manifest.language_model.backoff_weight",
                lo=0, lo_open=True, hi=1)
    else:
        _exact_keys(lm, {"type", "checkpoint"}, "manifest.language_model")
        _artifact(lm["checkpoint"], "manifest.language_model.checkpoint")

    st = data["stopping"]
    _exact_keys(st, set(DEFAULT_STOPPING), "manifest.stopping")
    _number(st["lam"], "manifest.stopping.lam", lo=0, hi=1)
    _number(st["theta"], "manifest.stopping.theta", lo=0, hi=1, lo_open=True, hi_open=True)
    _number(st["t_min"], "manifest.stopping.t_min", lo=0, integer=True)
    _number(st["t_max"], "manifest.stopping.t_max", lo=1, integer=True)
    if st["t_min"] > st["t_max"]:
        _fail("manifest.stopping.t_min", f"must be <= t_max ({st['t_max']}), got {st['t_min']}")
    _number(st["llr_clip"], "manifest.stopping.llr_clip", lo=0, lo_open=True)

    gate = data["quality_gate"]
    _exact_keys(gate, {"policy"}, "manifest.quality_gate")
    if gate["policy"] not in GATE_POLICIES:
        _fail("manifest.quality_gate.policy",
              f"must be one of {list(GATE_POLICIES)}, got {gate['policy']!r}")


def _read_json(path, label: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"{label} {path}: cannot read ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{label} {path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc


def load_characters_file(path) -> list[str]:
    data = _read_json(path, "characters file")
    chars = data.get("characters") if isinstance(data, dict) else data
    if not isinstance(chars, list) or len(chars) != N_CHARS:
        found = len(chars) if isinstance(chars, list) else "no list"
        raise ManifestError(f"characters file {path}: must list exactly {N_CHARS} characters, found {found}")
    if not all(isinstance(c, str) and c for c in chars):
        raise ManifestError(f"characters file {path}: every entry must be a non-empty string")
    if len(set(chars)) != N_CHARS:
        dupes = sorted({c for c in chars if chars.count(c) > 1})
        raise ManifestError(f"characters file {path}: duplicate entries {dupes}")
    return chars


def load_calibration_file(path) -> dict:
    """Platt (a, b) file written by the fit script; see docs/INTERFACES.md."""
    data = _read_json(path, "calibration file")
    where = f"calibration file {path}"
    _exact_keys(data, _CALIBRATION_REQUIRED, where, optional=_CALIBRATION_OPTIONAL)
    if data["format"] != CALIBRATION_FORMAT:
        _fail(f"{where}.format", f"must be {CALIBRATION_FORMAT!r}, got {data['format']!r}")
    _number(data["a"], f"{where}.a")
    _number(data["b"], f"{where}.b")
    if data["score_method"] != SCORE_METHOD:
        _fail(f"{where}.score_method", f"must be {SCORE_METHOD!r}, got {data['score_method']!r}")
    _sha256_field(data["classifier_sha256"], f"{where}.classifier_sha256")
    return data


@dataclass(frozen=True)
class SessionManifest:
    path: Path
    data: dict
    fingerprint: str

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.base_dir / candidate).resolve()

    @property
    def characters_path(self) -> Path:
        return self.resolve(self.data["characters"]["path"])

    @property
    def classifier_path(self) -> Path:
        return self.resolve(self.data["classifier"]["path"])

    @property
    def calibration_path(self) -> Path:
        return self.resolve(self.data["calibration"]["path"])

    @property
    def lm_config(self) -> dict:
        return json.loads(json.dumps(self.data["language_model"]))

    @property
    def corpus_path(self) -> Path | None:
        lm = self.data["language_model"]
        return self.resolve(lm["corpus"]["path"]) if lm["type"] == "ngram" else None

    @property
    def transformer_checkpoint_path(self) -> Path | None:
        lm = self.data["language_model"]
        return self.resolve(lm["checkpoint"]["path"]) if lm["type"] == "transformer" else None

    @property
    def stopping(self) -> dict:
        return dict(self.data["stopping"])

    @property
    def gate_policy(self) -> str:
        return self.data["quality_gate"]["policy"]

    @property
    def fingerprint_short(self) -> str:
        return short_fingerprint(self.fingerprint)


def verify_artifacts(manifest: SessionManifest) -> None:
    entries = [(key, manifest.data[key]) for key in ("characters", "classifier", "calibration")]
    lm_type = manifest.data["language_model"]["type"]
    if lm_type == "ngram":
        entries.append(("language_model.corpus", manifest.data["language_model"]["corpus"]))
    elif lm_type == "transformer":
        entries.append(("language_model.checkpoint", manifest.data["language_model"]["checkpoint"]))

    for label, entry in entries:
        resolved = manifest.resolve(entry["path"])
        if not resolved.is_file():
            raise ManifestError(f"{label} artifact missing: {resolved}")
        actual = sha256_file(resolved)
        if actual != entry["sha256"]:
            raise ManifestError(
                f"{label} artifact changed since the manifest was written: {resolved} "
                f"(manifest sha256 {entry['sha256'][:12]}, file sha256 {actual[:12]}). "
                "Restore the original file or regenerate the manifest.")

    load_characters_file(manifest.characters_path)
    calibration = load_calibration_file(manifest.calibration_path)
    classifier_sha = manifest.data["classifier"]["sha256"]
    if calibration["classifier_sha256"] != classifier_sha:
        raise ManifestError(
            f"calibration {manifest.calibration_path} was fitted for a different classifier "
            f"(sha256 {calibration['classifier_sha256'][:12]}) than the manifest's classifier "
            f"(sha256 {classifier_sha[:12]}). Platt (a, b) only applies to the scores it was fitted on.")


def load_manifest(path=DEFAULT_MANIFEST_PATH, *, verify_hashes: bool = True) -> SessionManifest:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ManifestError(
            f"session manifest not found: {path}. Create one with "
            "`python -m pipeline.fit_from_recording` (see docs/INTERFACES.md).")
    data = _read_json(path, "session manifest")
    validate_manifest_dict(data)
    manifest = SessionManifest(path=path, data=data, fingerprint=fingerprint_of(data))
    if verify_hashes:
        verify_artifacts(manifest)
    return manifest


def _relative(path, base_dir) -> str:
    return Path(os.path.relpath(Path(path).resolve(), Path(base_dir).resolve())).as_posix()


def artifact_entry(path, base_dir) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"artifact not found: {path}")
    return {"path": _relative(path, base_dir), "sha256": sha256_file(path)}


def build_manifest_dict(*, base_dir, characters_path, classifier_path, calibration_path,
                        language_model: dict | None = None, stopping: dict | None = None,
                        quality_gate: dict | None = None) -> dict:
    """
    base_dir must be the directory the manifest will be written to: artifact
    paths are stored relative to it.
    """
    lm = dict(language_model or {"type": "uniform"})
    if lm.get("type") == "ngram":
        corpus_path = lm.pop("corpus_path", None)
        if corpus_path is None:
            raise ManifestError("manifest.language_model: type 'ngram' requires corpus_path")
        lm = {**DEFAULT_NGRAM, **lm, "corpus": artifact_entry(corpus_path, base_dir)}
    elif lm.get("type") == "transformer":
        checkpoint_path = lm.pop("checkpoint_path", None)
        if checkpoint_path is None:
            raise ManifestError("manifest.language_model: type 'transformer' requires checkpoint_path")
        lm = {"type": "transformer", "checkpoint": artifact_entry(checkpoint_path, base_dir)}

    data = {
        "schema_version": SCHEMA_VERSION,
        "characters": artifact_entry(characters_path, base_dir),
        "classifier": artifact_entry(classifier_path, base_dir),
        "calibration": artifact_entry(calibration_path, base_dir),
        "language_model": lm,
        "stopping": {**DEFAULT_STOPPING, **(stopping or {})},
        "quality_gate": {**DEFAULT_QUALITY_GATE, **(quality_gate or {})},
    }
    validate_manifest_dict(data)
    return data


def write_manifest(data: dict, path) -> SessionManifest:
    """Atomic write (a crash never leaves a half-written manifest), then reload to verify."""
    validate_manifest_dict(data)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".session_manifest.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return load_manifest(path)


def assert_same_fingerprint(expected: str, observed: str | None, *, context: str) -> None:
    if not observed:
        raise ManifestMismatchError(
            f"{context}: the other process reported no manifest fingerprint, so there is no "
            "way to confirm it loaded the same models")
    if observed != expected:
        raise ManifestMismatchError(
            f"{context}: session manifest mismatch -- this process loaded "
            f"{short_fingerprint(expected)}, the other loaded {short_fingerprint(observed)}. "
            "Start both with the same session_manifest.json and unchanged model files.")
