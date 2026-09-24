"""
Experiment summary tests: pairing, accuracy/ITR/LM arithmetic on hand-built log records, the CLI on
tmp files, and one round trip through the real engine and the real JSON-lines formatter.
Every file is written under tmp_path.
"""

import functools
import json
import logging
import random
import sys

import numpy as np
import pytest

from pipeline import experiment_log as xlog
from pipeline import inference_service as svc
from pipeline import manifest as mf
from pipeline.epoching import EpochSpec
from stopping import summarize_experiment as sx
from stopping.bayesian_stopping import BayesianStoppingController
from stopping.replay import wolpaw_itr

sys.path.insert(0, str(mf.REPO_ROOT / "collection"))
from flash_sequence import CheckerboardConfig, build_flash_sequence  # noqa: E402

REPO_CHARACTERS = mf.REPO_ROOT / "collection" / "characters_th.json"


# ---------------------------------------------------------------------------
# record builders
# ---------------------------------------------------------------------------

def trial(tid, *, context="", target="ก", prior=0.5, rank=1, commit="ก", correct=True, n=10,
          reason="threshold_crossed", rejected=(), with_target=True):
    start = {"event": "trial_start", "trial_id": tid, "context": context, "lm_prior": [], "initial_posterior": []}
    if with_target:
        start.update(target_char_idx=0, target_char=target, target_prior_prob=prior, target_prior_rank=rank)
    else:
        start.update(target_char_idx=None, target_char=None)
    records = [start]
    records += [{"event": "flash_update", "trial_id": tid, "flash_idx": i, "t": i + 1} for i in range(min(n, 3))]
    records += [{"event": "flash_rejected", "trial_id": tid, "flash_idx": 90 + i, "reason": why, "channels": []}
                for i, why in enumerate(rejected)]
    if commit is not None:
        records.append({"event": "char_committed", "trial_id": tid, "char": commit, "reason": reason,
                        "n_flashes_used": n, "max_prob": 0.95,
                        "correct": correct if with_target else None})
    return records


def flat(*trials):
    return [r for t in trials for r in t]


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k,n,lo,hi", [(8, 10, 0.49016, 0.94332), (0, 10, 0.0, 0.27753), (36, 40, 0.76948, 0.96042)])
def test_wilson_interval_matches_reference_values(k, n, lo, hi):
    got = sx.wilson_interval(k, n)
    assert got == pytest.approx((lo, hi), abs=1e-4)
    assert sx.wilson_interval(0, 0) is None
    assert sx.wilson_interval(10, 10)[1] == pytest.approx(1.0)


def test_accuracy_flashes_and_itr_use_scored_characters_only():
    records = flat(trial(1, n=10), trial(2, n=20), trial(3, n=30), trial(4, n=40, commit="ข", correct=False),
                   trial(5, n=99, with_target=False))          # no known target: counted, not scored
    s = sx.summarize_inference(records, seconds_per_flash=0.1, overhead_s=1.5)
    assert (s["n_trials_started"], s["n_committed"], s["n_scored"], s["n_unscored"]) == (5, 5, 4, 1)
    assert (s["n_correct"], s["accuracy"], s["error_rate"]) == (3, 0.75, 0.25)
    assert s["mean_flashes"] == 25 and s["median_flashes"] == 25
    assert s["seconds_per_selection"] == pytest.approx(2.5)
    assert s["itr_bits_per_min"] == pytest.approx(wolpaw_itr(0.75, 72, 2.5))
    assert s["itr_with_overhead_bits_per_min"] == pytest.approx(wolpaw_itr(0.75, 72, 4.0))
    assert s["accuracy_ci95"] == pytest.approx(sx.wilson_interval(3, 4))


def test_n_classes_follows_the_logged_character_list():
    config = {"event": "service_config", "characters": [str(i) for i in range(10)], "manifest_fingerprint": "f" * 64,
              "manifest": {"language_model": {"type": "uniform"}, "stopping": {"theta": 0.9}}}
    s = sx.summarize_inference([config, *trial(1, n=10)], seconds_per_flash=0.1)
    assert s["n_chars"] == 10 and s["itr_bits_per_min"] == pytest.approx(wolpaw_itr(1.0, 10, 1.0))
    assert s["service_config"]["language_model"] == {"type": "uniform"}


def test_nothing_scored_gives_no_accuracy_or_itr():
    s = sx.summarize_inference(flat(trial(1, with_target=False)), seconds_per_flash=0.1)
    assert s["n_scored"] == 0 and s["accuracy"] is None and s["itr_bits_per_min"] is None
    assert s["accuracy_ci95"] is None
    assert "no committed character had a known target" in sx.format_report(
        {"session": None, "fit_report": None, "inference_log": "x", "unparseable_log_lines": 0, "inference": s})


def test_trials_are_paired_by_order_so_restarted_trial_ids_do_not_merge():
    records = flat(trial(1, n=10), trial(2, commit=None), trial(1, n=12), trial(2, n=14))   # ids restart
    s = sx.summarize_inference(records, seconds_per_flash=0.1)
    assert (s["n_trials_started"], s["n_committed"], s["n_without_commit"]) == (4, 3, 1)
    assert [r["n_flashes"] for r in s["trials"] if r["reason"]] == [10, 12, 14]


def test_lm_effect_compares_flashes_when_the_lm_ranked_the_target_first():
    records = flat(trial(1, prior=0.9, rank=1, n=8), trial(2, prior=0.7, rank=1, n=10),
                   trial(3, prior=0.02, rank=9, n=30), trial(4, prior=0.1, rank=3, n=20))
    lm = sx.summarize_inference(records, seconds_per_flash=0.1)["lm"]
    assert lm["n_with_target"] == 4 and lm["target_ranked_first"] == 2 and lm["target_in_top5"] == 3
    assert lm["mean_target_prior_prob"] == pytest.approx((0.9 + 0.7 + 0.02 + 0.1) / 4)
    assert lm["mean_flashes_when_ranked_first"] == 9 and lm["mean_flashes_when_not"] == 25
    assert lm["chance_prob"] == pytest.approx(1 / 72)


def test_reasons_and_rejected_flashes_are_counted():
    records = flat(trial(1, rejected=["quality_gate", "quality_gate"]),
                   trial(2, reason="max_flashes_reached", rejected=["epoch_incomplete"]),
                   trial(3, reason="trial_ended"))
    s = sx.summarize_inference(records, seconds_per_flash=0.1)
    assert s["reasons"] == {"threshold_crossed": 1, "max_flashes_reached": 1, "trial_ended": 1}
    assert s["rejected_flashes"] == {"quality_gate": 2, "epoch_incomplete": 1} and s["n_rejected_flashes"] == 3
    assert [r["n_rejected_flashes"] for r in s["trials"]] == [2, 1, 0]


def test_read_jsonl_skips_blank_bad_and_truncated_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"event": "a"}\n\n[1, 2]\nnot json\n{"event": "b"}\n{"event": "trunc', encoding="utf-8")
    records, bad = sx.read_jsonl(path)
    assert [r["event"] for r in records] == ["a", "b"] and bad == 3
    with pytest.raises(sx.SummaryError, match="cannot read"):
        sx.read_jsonl(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------------------
# files and CLI
# ---------------------------------------------------------------------------

def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def _experiment(tmp_path, *, with_fit_report=True):
    """A recording's sidecars: session record (against a real manifest), inference log, fit report."""
    models = tmp_path / "models"
    models.mkdir()
    clf = models / "clf.joblib"
    clf.write_bytes(b"fake-classifier")
    cal = models / "platt.json"
    cal.write_text(json.dumps({"format": mf.CALIBRATION_FORMAT, "a": 1.0, "b": 0.0, "score_method": mf.SCORE_METHOD,
                               "classifier_sha256": mf.sha256_file(clf)}))
    manifest = mf.write_manifest(mf.build_manifest_dict(
        base_dir=tmp_path, characters_path=REPO_CHARACTERS, classifier_path=clf, calibration_path=cal),
        tmp_path / "session_manifest.json")

    rec = tmp_path / "recordings" / "s1_20260923_101500.csv"
    rec.parent.mkdir()
    rec.write_text("timestamp\n", encoding="utf-8")
    characters = mf.load_characters_file(REPO_CHARACTERS)
    session = xlog.build_session_record(
        session_id="s1", demo=False, targets="กขค", characters=characters,
        settings={"repetitions": 5, "flash_ms": 80, "isi_ms": 90, "screen": 0, "windowed": False},
        recording=rec, manifest={"path": str(manifest.path), "fingerprint": manifest.fingerprint},
        started_utc="2026-09-23T10:15:00+00:00")
    xlog.write_json_atomic(xlog.finalize_session_record(session, ended_utc="2026-09-23T10:20:00+00:00",
                                                        message="Stimulus finished", level="info",
                                                        phase_at_end="running"),
                           xlog.session_record_path(rec))
    _write_jsonl(xlog.inference_log_path(rec), flat(trial(1, n=12), trial(2, n=20, commit="ข", correct=False)))
    if with_fit_report:
        xlog.write_json_atomic({
            "created_utc": "2026-09-22T09:00:00+00:00", "demo": False,
            "calibration_session": {"targets_text": "ทดสอบ"}, "recording": {"n_flash_markers": 600},
            "epochs": {"n_used": 590, "n_targets": 50, "n_non_targets": 540, "n_trials": 5, "rejected": {}},
            "cross_validation": {"n_splits": 5, "out_of_fold_auc": 0.83,
                                 "folds": [{"auc": 0.8}, {"auc": 0.87}, {"auc": None}]},
            "platt": {"a": 1.2, "b": -0.4, "ece": 0.03}}, xlog.fit_report_path(clf))
    return rec


def test_recording_finds_its_session_log_and_the_fit_report_of_its_manifest(tmp_path):
    rec = _experiment(tmp_path)
    exp = sx.load_experiment(recording=rec)
    assert exp["session"]["targets_text"] == "กขค" and exp["fit_report"]["platt"]["a"] == 1.2
    assert exp["inference"]["seconds_per_flash"] == pytest.approx(0.170)     # the session's flash_ms + isi_ms
    text = sx.format_report(exp)
    for expected in ("กขค", "ทดสอบ", "1/2 = 50.0%", "out-of-fold AUC 0.830", "folds 0.800-0.870",
                     "ended        Stimulus finished (reached: running)", "PER CHARACTER"):
        assert expected in text, expected


def test_overrides_and_the_missing_fit_report(tmp_path):
    rec = _experiment(tmp_path, with_fit_report=False)
    exp = sx.load_experiment(recording=rec, seconds_per_flash=0.124, overhead_s=2.0)
    assert exp["fit_report"] is None and exp["inference"]["seconds_per_flash"] == 0.124
    text = sx.format_report(exp)
    assert "no fit report found" in text and "with 2 s overhead" in text


def test_cli_prints_and_writes_json(tmp_path, capsys):
    rec = _experiment(tmp_path)
    out = tmp_path / "summary.json"
    assert sx.main(["--recording", str(rec), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "EXPERIMENT SUMMARY" in printed and str(out) in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["inference"]["n_scored"] == 2 and written["session"]["session_id"] == "s1"


def test_inference_log_alone_needs_no_session(tmp_path, capsys):
    log = tmp_path / "svc.log.jsonl"
    _write_jsonl(log, trial(1, n=10))
    assert sx.main(["--inference-log", str(log)]) == 0
    assert "no session record" in capsys.readouterr().out


@pytest.mark.parametrize("argv,message", [
    ([], "exactly one"),
    (["--recording", "a.csv", "--inference-log", "b.jsonl"], "exactly one"),
    (["--inference-log", "{tmp}/missing.jsonl"], "no inference log"),
    (["--inference-log", "{tmp}/empty.jsonl"], "no trial_start"),
    (["--inference-log", "{tmp}/empty.jsonl", "--seconds-per-flash", "0"], "positive"),
    (["--inference-log", "{tmp}/empty.jsonl", "--overhead-s", "-1"], ">= 0"),
    (["--inference-log", "{tmp}/ok.jsonl", "--fit-report", "{tmp}/nope.json"], "cannot read fit report"),
])
def test_cli_refuses_with_one_stderr_line(tmp_path, capsys, argv, message):
    (tmp_path / "empty.jsonl").write_text('{"event": "service_config"}\n', encoding="utf-8")
    _write_jsonl(tmp_path / "ok.jsonl", trial(1))
    assert sx.main([a.replace("{tmp}", str(tmp_path)) for a in argv]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and message in err and err.startswith("summarize_experiment: error:")


def test_a_recording_made_without_live_inference_says_so(tmp_path, capsys):
    rec = tmp_path / "calib.csv"
    rec.write_text("x\n", encoding="utf-8")
    assert sx.main(["--recording", str(rec)]) == 2
    assert "recorded without live inference" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# round trip: real engine -> real JSON-lines formatter -> summary
# ---------------------------------------------------------------------------

class _UniformLM:
    def get_prior(self, context):
        return np.full(72, 1 / 72)


def test_summary_of_a_log_written_by_the_real_engine(tmp_path):
    random.seed(0)
    characters = mf.load_characters_file(REPO_CHARACTERS)
    log_path = tmp_path / "svc.log.jsonl"
    logger = svc.setup_logging(log_path)
    try:
        scores = []
        engine = svc.InferenceEngine(
            controller_factory=functools.partial(BayesianStoppingController, n_chars=72, sigmoid_a=1.0,
                                                 sigmoid_b=0.0),
            language_model=_UniformLM(), scorer=lambda epoch: scores.pop(0), epoch_spec=EpochSpec(),
            gate_policy="skip", characters=characters, emit=lambda e: None, fingerprint="d" * 64,
            logger=logger.getChild("engine"))
        rng = np.random.default_rng(1)
        t0, soa = 100.0, 0.124
        targets = "กข"
        cursor, pos = t0 + 1.0, 0
        ts = t0 + np.arange(int(20 * 250)) / 250.0
        data = rng.normal(0, 10, size=(len(ts), 8))
        for tid, ch in enumerate(targets, start=1):
            tidx = characters.index(ch)
            engine.on_marker({"event": "trial_start", "trial_id": tid, "target_char_idx": tidx,
                              "target_char": ch}, cursor)
            groups = build_flash_sequence(CheckerboardConfig(n_chars=72, n_repetitions=1))
            for i, g in enumerate(groups):
                scores.append(float(rng.normal(4.0 if tidx in g else -4.0, 1.0)))
                engine.on_marker({"event": "flash", "trial_id": tid, "flash_idx": i, "group": list(g),
                                  "is_target": tidx in g}, cursor + 1.0 + i * soa)
            engine.on_marker({"event": "trial_end", "trial_id": tid}, cursor + 1.0 + len(groups) * soa)
            cursor += 1.0 + len(groups) * soa + 1.5
            stop = int(np.searchsorted(ts, cursor))
            while pos < stop:                     # decode this trial before queueing the next
                end = min(pos + 250, stop)
                engine.on_eeg_chunk(data[pos:end], ts[pos:end])
                pos = end
            scores.clear()                        # flashes after an early stop never reach the classifier
        for handler in logger.handlers:
            handler.flush()
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    exp = sx.load_experiment(inference_log=log_path, seconds_per_flash=soa)
    s = exp["inference"]
    assert engine.committed_text == "กข"
    assert (s["n_trials_started"], s["n_committed"], s["n_scored"], s["n_correct"]) == (2, 2, 2, 2)
    assert [r["context"] for r in s["trials"]] == ["", "ก"]
    assert all(r["target_prior_prob"] == pytest.approx(1 / 72, rel=1e-4) for r in s["trials"])   # uniform LM
    assert s["itr_bits_per_min"] == pytest.approx(wolpaw_itr(1.0, 72, s["mean_flashes"] * soa))
    assert "100.0%" in sx.format_report(exp)
