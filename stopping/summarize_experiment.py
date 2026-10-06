"""
Summarize one experiment from the logs it left behind (pipeline/experiment_log.py): the
calibration sentence, how the classifier was trained, and how every character was decoded.

    python -m stopping.summarize_experiment --recording collection/recordings/<session>.csv \\
        [--fit-report models/<id>_fit_report.json] [--seconds-per-flash 0.124] [--overhead-s 0] \\
        [--json results/summary.json] [--html results/report.html] [--benchmark PATH]
    python -m stopping.summarize_experiment --inference-log logs/inference_<ts>.log.jsonl

--html writes a self-contained, shareable HTML report: charts (accuracy vs. chance, per-character
flashes, LM-prior rank distribution), a findings list generated from the same numbers the text
report prints, and a benchmark table comparing this session's LM against stopping/
model_comparison_results.json (--benchmark to point elsewhere; the section is left out, not
guessed at, if neither is found). It needs no chart library and no network access -- every chart
is inline SVG built directly from the data above.

--recording finds <csv>.session.json and <csv>.inference.log.jsonl beside it, and, when the
session ran against a manifest, the fit report of that manifest's classifier. --inference-log
reads a service log on its own (no session or fit report unless --fit-report is given).

A BCI2000 recording (--recording <run>.dat, from CheckerboardSpeller) is read through
pipeline/bci2000_session.py, which turns the run's states and the speller's .applog into the same
records the inference service logs; its fit report is found by the classifier's model id in
models/. BCI2000 sessions are closed-loop, so they also report the flashes actually SHOWN per
character (more than the controller used: results lag their flashes by ~1.3 s) and an ITR on that.

What it reports, from the inference log:
  accuracy      over characters whose target the stimulus announced (the log's `correct`),
                with a 95% Wilson interval. Thirty to forty characters give a wide interval;
                report it. Characters with no known target are counted but not scored.
  ITR           Wolpaw bits/min via stopping.replay.wolpaw_itr, with seconds per selection
                = mean flashes x seconds per flash, the same basis as stopping.evaluate_recording
                (flash time only: no cue, no pause between characters). --overhead-s adds a
                per-selection overhead for a wall-clock figure.
  seconds/flash from the session's flash_ms + isi_ms, else --seconds-per-flash, else 0.124 s.
  LM effect     how much probability the language model gave the true character before any
                flash, and how many flashes characters took when the LM ranked them first
                versus not.

The log records the language model's prior at the start of every character, so this is a
description of what the LM and classifier did in that session, not a controlled comparison
of language models: to compare LMs, replay the same recording under each
(stopping.evaluate_recording).

Exit codes: 0 success, 2 refused or failed (one line on stderr).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from pipeline import experiment_log as xlog
from pipeline.manifest import ManifestError, load_manifest
from stopping.replay import wolpaw_itr

DEFAULT_SECONDS_PER_FLASH = 0.124   # 62 ms flash + 62 ms ISI, the stimulus defaults
DEFAULT_N_CHARS = 72
Z_95 = 1.959963984540054
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "stopping" / "model_comparison_results.json"


class SummaryError(Exception):
    """The logs are missing or unusable."""


def read_jsonl(path) -> tuple[list[dict], int]:
    """(records, n_unparseable_lines). A crashed service can leave a truncated last line."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc
    records, bad = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            bad += 1
    return records, bad


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a proportion; sensible at the small n and extreme p of a spelling test."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def collect_trials(records: list[dict]) -> tuple[dict | None, list[dict]]:
    """
    Pair the log's records into one dict per character: the trial_start, its flash records,
    and the commit. Pairing is by order, not trial_id alone, because trial ids restart when the
    stimulus restarts. A trial_start with no commit before the next trial_start is kept with
    `committed` None (the trial failed or the log was cut short).
    """
    config, trials, current = None, [], None

    def close():
        nonlocal current
        if current is not None:
            trials.append(current)
            current = None

    for r in records:
        kind = r.get("event")
        if kind == "service_config":
            config = r
        elif kind == "trial_start":
            close()
            current = {"start": r, "flashes": [], "rejected": [], "committed": None}
        elif current is not None and r.get("trial_id") == current["start"].get("trial_id"):
            if kind == "flash_update":
                current["flashes"].append(r)
            elif kind == "flash_rejected":
                current["rejected"].append(r)
            elif kind == "char_committed":
                current["committed"] = r
                close()
    close()
    return config, trials


def summarize_inference(records: list[dict], *, seconds_per_flash: float,
                        overhead_s: float = 0.0) -> dict:
    config, trials = collect_trials(records)
    n_chars = len(config["characters"]) if config and config.get("characters") else DEFAULT_N_CHARS

    rows = []
    for t in trials:
        s, c = t["start"], t["committed"]
        rows.append({
            "trial_id": s.get("trial_id"), "context": s.get("context", ""),
            "target_char": s.get("target_char"), "target_char_idx": s.get("target_char_idx"),
            "target_prior_prob": s.get("target_prior_prob"), "target_prior_rank": s.get("target_prior_rank"),
            "committed_char": c.get("char") if c else None,
            "correct": c.get("correct") if c else None,
            "n_flashes": c.get("n_flashes_used") if c else len(t["flashes"]),
            "reason": c.get("reason") if c else None,
            "final_max_prob": c.get("max_prob") if c else None,
            "n_flashes_presented": c.get("n_flashes_presented") if c else None,
            "n_rejected_flashes": len(t["rejected"]),
        })

    committed = [r for r in rows if r["reason"] is not None]
    scored = [r for r in committed if r["correct"] is not None]
    n_correct = sum(1 for r in scored if r["correct"])
    accuracy = n_correct / len(scored) if scored else None
    mean_flashes = _mean(r["n_flashes"] for r in scored)

    itr = itr_with_overhead = seconds_per_selection = None
    if scored and mean_flashes:
        seconds_per_selection = mean_flashes * seconds_per_flash
        itr = wolpaw_itr(accuracy, n_chars, seconds_per_selection)
        if overhead_s > 0:
            itr_with_overhead = wolpaw_itr(accuracy, n_chars, seconds_per_selection + overhead_s)

    # Closed-loop sessions (BCI2000) also record the flashes actually shown: the user's real cost.
    shown = [r["n_flashes_presented"] for r in scored]
    mean_shown = itr_shown = None
    if scored and all(v is not None for v in shown):
        mean_shown = _mean(shown)
        if mean_shown:
            itr_shown = wolpaw_itr(accuracy, n_chars, mean_shown * seconds_per_flash)

    rejected = Counter(r.get("reason") for t in trials for r in t["rejected"])

    # LM effect: only where the target is known and the LM's rank was logged
    known = [r for r in rows if r["target_prior_rank"] is not None]
    top1 = [r for r in known if r["target_prior_rank"] == 1]
    other = [r for r in known if r["target_prior_rank"] > 1]
    lm = {
        "n_with_target": len(known),
        "chance_prob": 1 / n_chars,
        "mean_target_prior_prob": _mean(r["target_prior_prob"] for r in known),
        "target_ranked_first": len(top1),
        "target_in_top5": sum(1 for r in known if r["target_prior_rank"] <= 5),
        "mean_flashes_when_ranked_first": _mean(r["n_flashes"] for r in top1 if r["reason"]),
        "mean_flashes_when_not": _mean(r["n_flashes"] for r in other if r["reason"]),
    }

    return {
        "n_chars": n_chars,
        "service_config": ({"manifest_fingerprint": config.get("manifest_fingerprint"),
                            "language_model": (config.get("manifest") or {}).get("language_model"),
                            "stopping": (config.get("manifest") or {}).get("stopping")}
                           if config else None),
        "n_trials_started": len(rows), "n_committed": len(committed), "n_scored": len(scored),
        "n_unscored": len(committed) - len(scored),
        "n_without_commit": len(rows) - len(committed),
        "n_correct": n_correct, "accuracy": accuracy,
        "accuracy_ci95": wilson_interval(n_correct, len(scored)),
        "error_rate": None if accuracy is None else 1 - accuracy,
        "mean_flashes": mean_flashes, "median_flashes": _median(r["n_flashes"] for r in scored),
        "seconds_per_flash": seconds_per_flash, "seconds_per_selection": seconds_per_selection,
        "itr_bits_per_min": itr, "overhead_s": overhead_s, "itr_with_overhead_bits_per_min": itr_with_overhead,
        "mean_flashes_presented": mean_shown, "itr_presented_bits_per_min": itr_shown,
        "reasons": dict(Counter(r["reason"] for r in committed)),
        "rejected_flashes": dict(rejected), "n_rejected_flashes": sum(rejected.values()),
        "lm": lm, "trials": rows,
    }


def build_findings(exp: dict) -> list[dict]:
    """
    Plain-language findings generated from the same numbers format_report() prints, each tagged
    "good" / "neutral" / "caution" for the HTML report's highlight styling. Every number here is
    computed above, not re-measured -- this is presentation, not new analysis.
    """
    inf = exp["inference"]
    session = exp["session"]
    findings = []

    def add(kind, text):
        findings.append({"kind": kind, "text": text})

    if session and session.get("demo"):
        add("caution", "This session is marked demo/synthetic data, not a real recording -- "
                       "every number below describes the synthetic session, not real performance.")

    n = inf["n_chars"]
    chance = 1 / n
    if inf["n_scored"]:
        lo, hi = inf["accuracy_ci95"]
        margin = hi - lo
        add("good" if inf["accuracy"] > chance * 3 else "neutral",
            f"Accuracy was {inf['accuracy']:.1%} ({inf['n_correct']}/{inf['n_scored']} characters), "
            f"{inf['accuracy'] / chance:.0f}x the {chance:.1%} chance rate for a {n}-character matrix.")
        if inf["n_scored"] < 30:
            add("caution", f"Only {inf['n_scored']} characters were scored, so the 95% confidence "
                          f"interval is wide ({lo:.1%} to {hi:.1%}, a {margin:.1%}-point range) -- "
                          "treat the accuracy figure as a rough estimate, not a precise one.")
        add("good" if inf["itr_bits_per_min"] > 0 else "caution",
            f"Information transfer rate was {inf['itr_bits_per_min']:.1f} bits/min "
            f"(Wolpaw; chance-level accuracy would give 0 bits/min regardless of speed).")

    lm = inf["lm"]
    if lm["n_with_target"]:
        rate = lm["target_ranked_first"] / lm["n_with_target"]
        chance_rate = 1 / n
        add("good" if rate > chance_rate * 3 else "neutral",
            f"The language model ranked the true next character 1st for {lm['target_ranked_first']}/"
            f"{lm['n_with_target']} characters ({rate:.1%}), versus {chance_rate:.1%} expected if it "
            "carried no information at all.")
        r1, ro = lm["mean_flashes_when_ranked_first"], lm["mean_flashes_when_not"]
        if r1 is not None and ro is not None and ro > 0:
            add("good" if r1 < ro else "neutral",
                f"Characters the LM ranked 1st took {r1:.1f} flashes on average to decode, versus "
                f"{ro:.1f} otherwise ({(1 - r1 / ro):+.0%} relative to the non-top-ranked group).")

    committed = inf["n_committed"]
    if committed:
        threshold = inf["reasons"].get("threshold_crossed", 0)
        add("good" if threshold / committed > 0.7 else "neutral",
            f"{threshold}/{committed} characters ({threshold / committed:.1%}) stopped because the "
            "posterior crossed the confidence threshold, rather than being forced by a flash-count "
            "limit or the trial ending.")
    if inf["n_rejected_flashes"]:
        add("caution", f"{inf['n_rejected_flashes']} flashes were rejected before reaching the "
                       f"controller: {', '.join(f'{k} {v}' for k, v in sorted(inf['rejected_flashes'].items()))}.")
    return findings


def load_benchmark(path=DEFAULT_BENCHMARK_PATH) -> dict | None:
    """stopping/model_comparison_results.json, if present -- see stopping/model_comparison.py."""
    return xlog.read_json_if_exists(path)


def find_fit_report(session: dict | None) -> Path | None:
    """The fit report of the classifier the session ran against, if the session named a manifest."""
    manifest = (session or {}).get("manifest") or {}
    if not manifest.get("path"):
        return None
    try:
        classifier = load_manifest(manifest["path"], verify_hashes=False).classifier_path
    except ManifestError:
        return None
    report = xlog.fit_report_path(classifier)
    return report if report.is_file() else None


def find_fit_report_for_model(model_id: str | None, models_dir=REPO_ROOT / "models") -> Path | None:
    """The fit report whose classifier checkpoint has this sha256 (a BCI2000 run's RiemannModelID)."""
    if not model_id:
        return None
    for path in sorted(Path(models_dir).glob("*_fit_report.json")):
        report = xlog.read_json_if_exists(path) or {}
        if ((report.get("artifacts") or {}).get("classifier") or {}).get("sha256") == model_id:
            return path
    return None


def load_bci2000_experiment(recording, *, fit_report=None, seconds_per_flash=None, overhead_s: float = 0.0,
                            models_dir=REPO_ROOT / "models") -> dict:
    from pipeline.bci2000_dat import DatError, read_dat
    from pipeline.bci2000_recording import RecordingError
    from pipeline import bci2000_session as bs
    from pipeline.fit_from_recording import dat_synthetic_reason

    try:
        dat = read_dat(recording)
        applog = bs.find_applog(recording)
        records = bs.session_records(dat, applog.read_text(encoding="utf-8", errors="replace") if applog else None)
        spf = seconds_per_flash or bs.seconds_per_flash(dat) or DEFAULT_SECONDS_PER_FLASH
    except (DatError, RecordingError, bs.SessionError) as exc:
        raise SummaryError(str(exc)) from exc
    summary = summarize_inference(records, seconds_per_flash=spf, overhead_s=overhead_s)
    if summary["n_trials_started"] == 0:
        raise SummaryError(f"{recording}: no selections in this run")
    synthetic = dat_synthetic_reason(dat)
    targets = [r["target_char"] for r in records if r.get("event") == "trial_start" and r.get("target_char")]
    p = lambda name: dat.param_str(name) if name in dat.params else None  # noqa: E731
    session = {"session_id": Path(recording).stem,
               "kind": "BCI2000 " + ("copy spelling" if p("InterpretMode") == "2" else "free spelling"),
               "demo": synthetic is not None, "started_utc": p("StorageTime"), "ended_utc": None,
               "targets_text": p("TextToSpell"), "target_sequence": targets,
               "synthetic_reason": synthetic, "applog": applog.as_posix() if applog else None}
    model_id = (records[0].get("manifest_fingerprint") if records else None)
    report_path = Path(fit_report) if fit_report else find_fit_report_for_model(model_id, models_dir)
    report = xlog.read_json_if_exists(report_path) if report_path else None
    if fit_report and report is None:
        raise SummaryError(f"cannot read fit report {fit_report}")
    return {"session": session, "fit_report": report,
            "fit_report_path": report_path.as_posix() if report_path else None,
            "inference_log": Path(recording).as_posix(), "unparseable_log_lines": 0, "inference": summary}


def load_experiment(*, recording=None, inference_log=None, fit_report=None,
                    seconds_per_flash=None, overhead_s: float = 0.0) -> dict:
    if (recording is None) == (inference_log is None):
        raise SummaryError("give exactly one of --recording and --inference-log")
    if recording is not None and Path(recording).suffix.lower() == ".dat":
        return load_bci2000_experiment(recording, fit_report=fit_report, seconds_per_flash=seconds_per_flash,
                                       overhead_s=overhead_s)
    session = None
    if recording is not None:
        session = xlog.read_json_if_exists(xlog.session_record_path(recording))
        inference_log = xlog.inference_log_path(recording)
    if not Path(inference_log).is_file():
        raise SummaryError(f"no inference log at {inference_log}"
                           + (" (was this session recorded without live inference?)" if recording else ""))

    settings = (session or {}).get("settings") or {}
    if seconds_per_flash is None:
        if isinstance(settings.get("flash_ms"), (int, float)) and isinstance(settings.get("isi_ms"), (int, float)):
            seconds_per_flash = (settings["flash_ms"] + settings["isi_ms"]) / 1000.0
        else:
            seconds_per_flash = DEFAULT_SECONDS_PER_FLASH

    records, n_bad = read_jsonl(inference_log)
    summary = summarize_inference(records, seconds_per_flash=seconds_per_flash, overhead_s=overhead_s)
    if summary["n_trials_started"] == 0:
        raise SummaryError(f"{inference_log} has no trial_start records; nothing to summarize "
                           "(a log from before experiment logging, or a session with no trials?)")

    report_path = Path(fit_report) if fit_report else find_fit_report(session)
    report = xlog.read_json_if_exists(report_path) if report_path else None
    if fit_report and report is None:
        raise SummaryError(f"cannot read fit report {fit_report}")
    return {"session": session, "fit_report": report,
            "fit_report_path": report_path.as_posix() if report_path else None,
            "inference_log": Path(inference_log).as_posix(), "unparseable_log_lines": n_bad,
            "inference": summary}


def _fmt(value, spec=".3f", missing="n/a"):
    return missing if value is None else format(value, spec)


def format_report(exp: dict) -> str:
    out = []
    session, report, inf = exp["session"], exp["fit_report"], exp["inference"]

    out.append("EXPERIMENT SUMMARY")
    if session:
        tag = "demo, synthetic data" if session.get("demo") else "real"
        out.append(f"  session      {session.get('session_id')} ({session.get('kind')}, {tag})  "
                   f"{session.get('started_utc')} -> {session.get('ended_utc') or 'not finalized'}")
        out.append(f"  spelled      {session.get('targets_text')}  ({len(session.get('target_sequence') or [])} characters)")
        end = session.get("end") or {}
        if end:
            out.append(f"  ended        {end.get('message')} (reached: {end.get('phase_at_end')})")
    else:
        out.append("  session      no session record (log read on its own)")
    out.append(f"  inference log  {exp['inference_log']}"
               + (f"  [{exp['unparseable_log_lines']} unparseable line(s) skipped]" if exp["unparseable_log_lines"] else ""))

    cfg = inf["service_config"]
    if cfg:
        lm = (cfg.get("language_model") or {}).get("type")
        st = cfg.get("stopping") or {}
        out.append(f"  models       LM {lm}; theta {st.get('theta')}, t_min {st.get('t_min')}, t_max {st.get('t_max')}, "
                   f"lam {st.get('lam')}; manifest {(cfg.get('manifest_fingerprint') or '')[:12]}")

    out.append("")
    out.append("CLASSIFIER TRAINING")
    if report:
        cal = report.get("calibration_session") or {}
        cv, platt, ep = report["cross_validation"], report["platt"], report["epochs"]
        out.append(f"  calibration  {cal.get('targets_text', 'sentence not recorded')}"
                   f"  ({report['recording']['n_flash_markers']} flashes, {ep['n_trials']} trials)")
        folds = [f["auc"] for f in cv["folds"] if f.get("auc") is not None]
        fold_txt = f"  folds {min(folds):.3f}-{max(folds):.3f}" if folds else ""
        out.append(f"  epochs       {ep['n_used']} used ({ep['n_targets']} target, {ep['n_non_targets']} non-target); "
                   f"rejected {ep['rejected'] or 'none'}")
        out.append(f"  out-of-fold AUC {cv['out_of_fold_auc']:.3f} ({cv['n_splits']} folds by trial){fold_txt}")
        out.append(f"  Platt        a={platt['a']:.4f} b={platt['b']:.4f}  ECE={platt['ece']:.4f}")
        out.append(f"  fitted       {report['created_utc']}  {'(DEMO model)' if report.get('demo') else ''}".rstrip())
    else:
        out.append("  no fit report found (pass --fit-report to include one)")

    out.append("")
    out.append("INFERENCE")
    out.append(f"  characters   {inf['n_trials_started']} started, {inf['n_committed']} committed, "
               f"{inf['n_scored']} scored against a known target"
               + (f", {inf['n_unscored']} committed without a known target" if inf["n_unscored"] else "")
               + (f", {inf['n_without_commit']} never committed" if inf["n_without_commit"] else ""))
    if inf["n_scored"]:
        lo, hi = inf["accuracy_ci95"]
        out.append(f"  accuracy     {inf['n_correct']}/{inf['n_scored']} = {inf['accuracy']:.1%}   "
                   f"(95% CI {lo:.1%} to {hi:.1%});  error rate {inf['error_rate']:.1%}")
        out.append(f"  flashes      mean {inf['mean_flashes']:.1f}, median {_fmt(inf['median_flashes'], '.1f')} per character "
                   f"({inf['seconds_per_flash'] * 1000:.0f} ms per flash -> {inf['seconds_per_selection']:.1f} s per selection)")
        out.append(f"  ITR          {inf['itr_bits_per_min']:.1f} bits/min (Wolpaw, N={inf['n_chars']}, flash time only)"
                   + (f";  {inf['itr_with_overhead_bits_per_min']:.1f} bits/min with {inf['overhead_s']:g} s overhead"
                      if inf["itr_with_overhead_bits_per_min"] is not None else ""))
        if inf.get("itr_presented_bits_per_min") is not None:
            out.append(f"  shown        mean {inf['mean_flashes_presented']:.1f} flashes per character actually shown "
                       f"(closed loop; results lag ~1.3 s) -> ITR {inf['itr_presented_bits_per_min']:.1f} bits/min")
    else:
        out.append("  accuracy     n/a: no committed character had a known target")
    reasons = ", ".join(f"{k} {v}" for k, v in sorted(inf["reasons"].items())) or "none"
    out.append(f"  stopped by   {reasons}")
    if inf["n_rejected_flashes"]:
        out.append(f"  rejected     {inf['n_rejected_flashes']} flashes: "
                   + ", ".join(f"{k} {v}" for k, v in sorted(inf["rejected_flashes"].items(), key=str)))

    lm = inf["lm"]
    out.append("")
    out.append("LANGUAGE MODEL PRIOR (before any flash)")
    if lm["n_with_target"]:
        out.append(f"  P(true char) mean {lm['mean_target_prior_prob']:.3f}  (uniform would be {lm['chance_prob']:.3f})")
        out.append(f"  true char ranked 1st by the LM: {lm['target_ranked_first']}/{lm['n_with_target']};  "
                   f"in its top 5: {lm['target_in_top5']}/{lm['n_with_target']}")
        out.append(f"  mean flashes when ranked 1st {_fmt(lm['mean_flashes_when_ranked_first'], '.1f')}"
                   f"  vs otherwise {_fmt(lm['mean_flashes_when_not'], '.1f')}")
    else:
        out.append("  no character with a known target")

    out.append("")
    out.append("PER CHARACTER")
    out.append(f"  {'#':>3}  {'context':<10} {'target':<6} {'got':<5} {'ok':<3} {'flashes':>7}  {'P_LM(tgt)':>9} {'rank':>4}  reason")
    for i, r in enumerate(inf["trials"], start=1):
        ok = "-" if r["correct"] is None else ("yes" if r["correct"] else "NO")
        out.append(f"  {i:>3}  {r['context'][-8:]:<10} {r['target_char'] or '?':<6} {r['committed_char'] or '-':<5} {ok:<3} "
                   f"{r['n_flashes']:>7}  {_fmt(r['target_prior_prob'], '9.4f', ' ' * 8 + '-')} "
                   f"{'-' if r['target_prior_rank'] is None else r['target_prior_rank']:>4}  {r['reason'] or 'no commit'}")
    return "\n".join(out)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _svg_hbar_with_reference(rows, *, reference=None, reference_label="", width=640, bar_h=32,
                             gap=14, label_w=210, value_fmt=lambda v: f"{v:.1%}") -> str:
    """rows: [(label, value, color), ...]. reference: one value (e.g. chance level) marked with a
    dashed line across every bar, so each bar is read against the same baseline at a glance."""
    if not rows:
        return "<p class='muted'>No data.</p>"
    top_pad = 18 if reference is not None else 4
    height = top_pad + len(rows) * (bar_h + gap)
    max_val = max([v for _, v, _ in rows] + ([reference] if reference is not None else [0.0])) or 1.0
    plot_w = width - label_w - 70

    def x(v):
        return label_w + (v / max_val) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
             f'role="img" aria-label="bar chart">']
    for i, (label, value, color) in enumerate(rows):
        y = top_pad + gap + i * (bar_h + gap) - gap
        bw = max(2.0, x(value) - label_w)
        parts.append(f'<text x="{label_w - 10}" y="{y + bar_h * 0.66:.1f}" text-anchor="end" '
                     f'class="bar-label">{_esc(label)}</text>')
        parts.append(f'<rect x="{label_w}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h}" rx="5" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{x(value) + 8:.1f}" y="{y + bar_h * 0.66:.1f}" '
                     f'class="bar-value">{_esc(value_fmt(value))}</text>')
    if reference is not None:
        rx = x(reference)
        parts.append(f'<line x1="{rx:.1f}" y1="2" x2="{rx:.1f}" y2="{height - 2}" stroke="#94a3b8" '
                     f'stroke-width="1.5" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{rx:.1f}" y="13" text-anchor="middle" '
                     f'class="bar-ref">{_esc(reference_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_per_character_chart(trials, *, width=720, height=200, bar_gap=3) -> str:
    """One bar per committed character: height = flashes used, color = correct (teal) / incorrect
    (red) / no known target (gray). A dashed line marks the mean. Hover a bar for its detail."""
    rows = [t for t in trials if t["reason"] is not None]
    if not rows:
        return "<p class='muted'>No committed characters to chart.</p>"
    n = len(rows)
    max_flashes = max(r["n_flashes"] for r in rows) or 1
    plot_h = height - 24
    bar_w = max(3.0, (width - 36) / n - bar_gap)
    mean_flashes = sum(r["n_flashes"] for r in rows) / n

    def color(r):
        if r["correct"] is True:
            return "#1e7f72"
        if r["correct"] is False:
            return "#b91c1c"
        return "#94a3b8"

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" '
             f'role="img" aria-label="flashes per character">']
    for i, r in enumerate(rows):
        x0 = 32 + i * (bar_w + bar_gap)
        h = (r["n_flashes"] / max_flashes) * plot_h
        y0 = plot_h - h + 8
        title = f"#{i + 1} {r['target_char'] or '?'} -> {r['committed_char'] or '-'}, {r['n_flashes']} flashes"
        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                     f'fill="{color(r)}" rx="1.5"><title>{_esc(title)}</title></rect>')
    mean_y = plot_h - (mean_flashes / max_flashes) * plot_h + 8
    parts.append(f'<line x1="0" y1="{mean_y:.1f}" x2="{width}" y2="{mean_y:.1f}" stroke="#475569" '
                 f'stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{width - 4}" y="{mean_y - 4:.1f}" text-anchor="end" '
                 f'class="bar-ref">mean {mean_flashes:.1f}</text>')
    parts.append(f'<line x1="30" y1="{plot_h + 8}" x2="{width}" y2="{plot_h + 8}" stroke="#cbd5e1"/>')
    parts.append(f'<text x="2" y="{plot_h + 8 + 11}" class="axis-label">1</text>')
    parts.append(f'<text x="2" y="16" class="axis-label">{max_flashes}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _benchmark_table_html(benchmark: dict, this_lm_type: str | None) -> str:
    models = benchmark.get("models", {})
    if not models:
        return ""
    # "transformer" manifests use the production checkpoint (new_transformer, per CLAUDE.md);
    # old_transformer is kept in the comparison only for reference and is never what's running.
    highlight_key = {"transformer": "new_transformer", "ngram": "ngram"}.get(this_lm_type)
    rows = []
    for key, m in models.items():
        is_this = key == highlight_key
        params = f"{m['n_params']:,}" if m.get("n_params") is not None else (
            f"{m['n_context_entries']:,} entries" if m.get("n_context_entries") is not None else "n/a")
        rows.append(
            f'<tr class="{"this-session" if is_this else ""}">'
            f'<td>{_esc(m.get("label", key))}{" &larr; this session" if is_this else ""}</td>'
            f'<td>{params}</td>'
            f'<td>{m["top1_accuracy"]:.1%}</td><td>{m["top3_accuracy"]:.1%}</td>'
            f'<td>{m["mean_surprisal_bits"]:.2f} bits</td>'
            f'<td>{m["latency_ms"]["mean_ms"]:.3g} ms</td></tr>')
    return f"""
    <table class="bench">
      <thead><tr><th>Candidate</th><th>Size</th><th>Top-1</th><th>Top-3</th>
        <th>Mean surprisal</th><th>Mean latency</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    <p class="muted">From <code>stopping/model_comparison.py</code>: a fixed 16-word Thai
    completion test, CPU-only. This measures each candidate <em>language model's own</em>
    next-character prediction quality, not this session's overall decode accuracy above --
    the two numbers are not directly comparable, only the LM choice is shared between them.</p>
    """


def build_html_report(exp: dict, *, benchmark: dict | None = None) -> str:
    inf, session, report = exp["inference"], exp["session"], exp["fit_report"]
    findings = build_findings(exp)
    n = inf["n_chars"]
    chance = 1 / n

    title = "Experiment Report"
    if session:
        title = f"{session.get('session_id', 'Session')} report"

    header_bits = []
    if session:
        tag = "DEMO / synthetic data" if session.get("demo") else "real session"
        header_bits.append(f"<span class='pill {'pill-demo' if session.get('demo') else 'pill-real'}'>{_esc(tag)}</span>")
        header_bits.append(f"<span>{_esc(session.get('kind', ''))}</span>")
        header_bits.append(f"<span>{_esc(session.get('started_utc', ''))}</span>")
        if session.get("targets_text"):
            header_bits.append(f"<span>spelled &ldquo;{_esc(session['targets_text'])}&rdquo;</span>")
    header_meta = " &middot; ".join(header_bits) if header_bits else ""

    findings_html = "".join(
        f'<li class="finding finding-{f["kind"]}">{_esc(f["text"])}</li>' for f in findings
    ) or "<li class='muted'>Not enough data to generate findings.</li>"

    accuracy_chart = lm_chart = per_char_chart = ""
    if inf["n_scored"]:
        accuracy_chart = _svg_hbar_with_reference(
            [("This session", inf["accuracy"], "#1e7f72")],
            reference=chance, reference_label=f"chance ({chance:.1%})",
            value_fmt=lambda v: f"{v:.1%}")
    if inf["lm"]["n_with_target"]:
        rate = inf["lm"]["target_ranked_first"] / inf["lm"]["n_with_target"]
        lm_chart = _svg_hbar_with_reference(
            [("Ranked 1st by LM", rate, "#1e7f72")],
            reference=chance, reference_label=f"chance ({chance:.1%})",
            value_fmt=lambda v: f"{v:.1%}")
    per_char_chart = _svg_per_character_chart(inf["trials"])

    bench_html = ""
    if benchmark:
        lm_type = ((inf.get("service_config") or {}).get("language_model") or {}).get("type")
        bench_html = _benchmark_table_html(benchmark, lm_type)

    fit_html = ""
    if report:
        cv, platt, ep = report["cross_validation"], report["platt"], report["epochs"]
        cal = report.get("calibration_session") or {}
        fit_html = f"""
        <table class="kv">
          <tr><th>Calibration sentence</th><td>{_esc(cal.get('targets_text', 'not recorded'))}</td></tr>
          <tr><th>Epochs used</th><td>{ep['n_used']} ({ep['n_targets']} target, {ep['n_non_targets']} non-target)</td></tr>
          <tr><th>Out-of-fold AUC</th><td>{cv['out_of_fold_auc']:.3f} ({cv['n_splits']} folds by trial)</td></tr>
          <tr><th>Platt calibration</th><td>a={platt['a']:.4f} b={platt['b']:.4f} ECE={platt['ece']:.4f}</td></tr>
        </table>"""

    rows_html = ""
    for i, r in enumerate(inf["trials"], start=1):
        ok = "-" if r["correct"] is None else ("&#10003;" if r["correct"] else "&#10007;")
        ok_class = "ok-na" if r["correct"] is None else ("ok-yes" if r["correct"] else "ok-no")
        rows_html += (f"<tr><td>{i}</td><td>{_esc(r['context'][-10:])}</td>"
                     f"<td>{_esc(r['target_char'] or '?')}</td><td>{_esc(r['committed_char'] or '-')}</td>"
                     f"<td class='{ok_class}'>{ok}</td><td>{r['n_flashes']}</td>"
                     f"<td>{_fmt(r['target_prior_prob'], '.3f', '-')}</td>"
                     f"<td>{'-' if r['target_prior_rank'] is None else r['target_prior_rank']}</td>"
                     f"<td>{_esc(r['reason'] or 'no commit')}</td></tr>")

    stats_html = ""
    if inf["n_scored"]:
        lo, hi = inf["accuracy_ci95"]
        stats_html = f"""
        <div class="stats">
          <div class="stat"><div class="stat-value">{inf['accuracy']:.1%}</div>
            <div class="stat-label">accuracy (95% CI {lo:.1%}-{hi:.1%})</div></div>
          <div class="stat"><div class="stat-value">{inf['itr_bits_per_min']:.1f}</div>
            <div class="stat-label">bits/min (Wolpaw ITR)</div></div>
          <div class="stat"><div class="stat-value">{inf['mean_flashes']:.1f}</div>
            <div class="stat-label">mean flashes/character</div></div>
          <div class="stat"><div class="stat-value">{inf['n_scored']}</div>
            <div class="stat-label">characters scored</div></div>
        </div>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ color-scheme:light; --bg:#f8fafc; --surface:#ffffff; --ink:#0f172a; --muted:#64748b;
           --border:#e2e8f0; --good:#1e7f72; --good-bg:#e6f4f1; --caution:#b45309;
           --caution-bg:#fef3c7; --neutral-bg:#eef2f7; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ color-scheme:dark; --bg:#0b1220; --surface:#111827; --ink:#e5e7eb; --muted:#94a3b8;
             --border:#1f2937; --good:#34d399; --good-bg:#0f2e28; --caution:#fbbf24;
             --caution-bg:#3a2a08; --neutral-bg:#1a2333; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--ink); font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;
          margin:0; padding:24px 16px 60px; }}
  .wrap {{ max-width:920px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 4px; }}
  h2 {{ font-size:17px; margin:32px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
  .meta {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .pill {{ display:inline-block; padding:2px 9px; border-radius:99px; font-size:11px; font-weight:600;
           text-transform:uppercase; letter-spacing:.03em; }}
  .pill-demo {{ background:var(--caution-bg); color:var(--caution); }}
  .pill-real {{ background:var(--good-bg); color:var(--good); }}
  .card {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:18px 20px;
           margin-bottom:8px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:14px; margin:14px 0 4px; }}
  .stat {{ flex:1 1 160px; background:var(--surface); border:1px solid var(--border); border-radius:10px;
           padding:14px 16px; }}
  .stat-value {{ font-size:24px; font-weight:700; }}
  .stat-label {{ font-size:12px; color:var(--muted); margin-top:2px; }}
  ul.findings {{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:8px; }}
  .finding {{ padding:10px 14px; border-radius:8px; font-size:14px; line-height:1.5; }}
  .finding-good {{ background:var(--good-bg); border-left:3px solid var(--good); }}
  .finding-neutral {{ background:var(--neutral-bg); border-left:3px solid var(--muted); }}
  .finding-caution {{ background:var(--caution-bg); border-left:3px solid var(--caution); color:var(--ink); }}
  .bar-label {{ font-size:13px; fill:var(--ink); }}
  .bar-value {{ font-size:13px; fill:var(--ink); font-weight:600; }}
  .bar-ref {{ font-size:11px; fill:var(--muted); }}
  .axis-label {{ font-size:10px; fill:var(--muted); }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.kv th {{ text-align:left; color:var(--muted); font-weight:500; padding:5px 12px 5px 0; white-space:nowrap; }}
  table.kv td {{ padding:5px 0; }}
  table.bench, table.rows {{ margin-top:8px; }}
  table.bench th, table.bench td, table.rows th, table.rows td {{
    border-bottom:1px solid var(--border); padding:7px 10px; text-align:left; }}
  table.bench tr.this-session td {{ font-weight:600; }}
  tr.ok-yes, .ok-yes {{ color:var(--good); }}
  tr.ok-no, .ok-no {{ color:#b91c1c; }}
  .ok-na {{ color:var(--muted); }}
  .muted {{ color:var(--muted); font-size:13px; }}
  code {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; }}
  .scroll {{ overflow-x:auto; }}
</style></head>
<body><div class="wrap">
  <h1>{_esc(title)}</h1>
  <div class="meta">{header_meta}</div>

  <h2>Key findings</h2>
  <div class="card"><ul class="findings">{findings_html}</ul></div>
  {stats_html}

  {"<h2>Accuracy vs. chance</h2><div class='card'>" + accuracy_chart + "</div>" if accuracy_chart else ""}
  {"<h2>Language model: was the true character ranked 1st?</h2><div class='card'>" + lm_chart + "</div>" if lm_chart else ""}
  {"<h2>Flashes per character</h2><div class='card'>" + per_char_chart + "<p class='muted'>Teal = correct, red = incorrect, gray = no known target. Dashed line = mean.</p></div>" if per_char_chart else ""}
  {"<h2>Classifier training</h2><div class='card'>" + fit_html + "</div>" if fit_html else ""}
  {"<h2>Compared with other language models</h2><div class='card scroll'>" + bench_html + "</div>" if bench_html else ""}

  <h2>Per-character detail</h2>
  <div class="card scroll">
  <table class="rows">
    <thead><tr><th>#</th><th>Context</th><th>Target</th><th>Got</th><th>OK</th><th>Flashes</th>
      <th>P<sub>LM</sub>(target)</th><th>Rank</th><th>Stopped by</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>

  <p class="muted" style="margin-top:28px">
    Generated by <code>stopping/summarize_experiment.py</code> from
    <code>{_esc(exp['inference_log'])}</code>. Accuracy and LM-rank figures cover only characters
    whose target the stimulus announced; the language model's own training loss and this session's
    decode accuracy are different measurements and are not compared directly above.
  </p>
</div></body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stopping.summarize_experiment",
                                     description="Summarize an experiment from its logs.")
    parser.add_argument("--recording", help="recording CSV (finds its session record and inference log), "
                                            "or a BCI2000 .dat from CheckerboardSpeller")
    parser.add_argument("--inference-log", help="an inference service log, read on its own")
    parser.add_argument("--fit-report", help="fit report to include (default: found via the session's manifest)")
    parser.add_argument("--seconds-per-flash", type=float, default=None,
                        help="default: the session's flash_ms + isi_ms, else 0.124")
    parser.add_argument("--overhead-s", type=float, default=0.0,
                        help="extra seconds per selection (cue, pause) for a second, wall-clock ITR")
    parser.add_argument("--json", dest="json_out", help="also write the full summary here")
    parser.add_argument("--html", dest="html_out",
                        help="also write a chart+table HTML report here (see module docstring)")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH),
                        help="LM comparison data for the --html report's benchmark table "
                             f"(default: {DEFAULT_BENCHMARK_PATH.name}; omitted if not found)")
    args = parser.parse_args(argv)
    try:
        if args.seconds_per_flash is not None and args.seconds_per_flash <= 0:
            raise SummaryError("--seconds-per-flash must be positive")
        if args.overhead_s < 0:
            raise SummaryError("--overhead-s must be >= 0")
        exp = load_experiment(recording=args.recording, inference_log=args.inference_log,
                              fit_report=args.fit_report, seconds_per_flash=args.seconds_per_flash,
                              overhead_s=args.overhead_s)
        print(format_report(exp))
        if args.json_out:
            xlog.write_json_atomic(exp, args.json_out)
            print(f"\nsummary written to {args.json_out}")
        if args.html_out:
            benchmark = load_benchmark(args.benchmark)
            html = build_html_report(exp, benchmark=benchmark)
            Path(args.html_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.html_out).write_text(html, encoding="utf-8")
            print(f"HTML report written to {args.html_out}"
                 + ("" if benchmark else "  (no benchmark data found; that section was left out)"))
    except (SummaryError, OSError) as exc:
        print(f"summarize_experiment: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
