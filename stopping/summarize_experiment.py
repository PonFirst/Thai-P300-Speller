"""
Summarize one experiment from the logs it left behind (pipeline/experiment_log.py): the
calibration sentence, how the classifier was trained, and how every character was decoded.

    python -m stopping.summarize_experiment --recording collection/recordings/<session>.csv \\
        [--fit-report models/<id>_fit_report.json] [--seconds-per-flash 0.124] [--overhead-s 0] \\
        [--json results/summary.json]
    python -m stopping.summarize_experiment --inference-log logs/inference_<ts>.log.jsonl

--recording finds <csv>.session.json and <csv>.inference.log.jsonl beside it, and, when the
session ran against a manifest, the fit report of that manifest's classifier. --inference-log
reads a service log on its own (no session or fit report unless --fit-report is given).

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
        "reasons": dict(Counter(r["reason"] for r in committed)),
        "rejected_flashes": dict(rejected), "n_rejected_flashes": sum(rejected.values()),
        "lm": lm, "trials": rows,
    }


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


def load_experiment(*, recording=None, inference_log=None, fit_report=None,
                    seconds_per_flash=None, overhead_s: float = 0.0) -> dict:
    if (recording is None) == (inference_log is None):
        raise SummaryError("give exactly one of --recording and --inference-log")
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stopping.summarize_experiment",
                                     description="Summarize an experiment from its logs.")
    parser.add_argument("--recording", help="recording CSV; finds its session record and inference log")
    parser.add_argument("--inference-log", help="an inference service log, read on its own")
    parser.add_argument("--fit-report", help="fit report to include (default: found via the session's manifest)")
    parser.add_argument("--seconds-per-flash", type=float, default=None,
                        help="default: the session's flash_ms + isi_ms, else 0.124")
    parser.add_argument("--overhead-s", type=float, default=0.0,
                        help="extra seconds per selection (cue, pause) for a second, wall-clock ITR")
    parser.add_argument("--json", dest="json_out", help="also write the full summary here")
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
    except (SummaryError, OSError) as exc:
        print(f"summarize_experiment: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
