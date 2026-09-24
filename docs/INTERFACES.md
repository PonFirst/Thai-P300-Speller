# Interfaces between recorder, inference service, and GUI

The code is the source of truth. This page says where each contract lives, what was decided, and why.

| Contract | Defined in | Checked by |
|---|---|---|
| Stream identities, channel order, sample rate, quality thresholds, inlet timestamp processing | `pipeline/constants.py` | every consumer imports them |
| Marker JSON (stimulus → recorder, service) | `collection/checkerboard_stimulus.py` | recorder and service parsing |
| Recorder CSV | `collection/record_session.py` (module docstring) | fit script loader |
| `P300Inference` events (service → GUI) | `pipeline/events.py` | `validate_event()` on both ends |
| `session_manifest.json` | `pipeline/manifest.py` | `load_manifest()` in every process |
| Quality levels and the quality gate | `pipeline/quality.py` | GUI scalp map and service gate share it |
| Epoch cutting and preprocessing | `pipeline/epoching.py` | fit script and service share it |
| Experiment logs: file names, session record, fit report | `pipeline/experiment_log.py` | GUI, fit script, `tests/test_experiment_log.py` |

## LSL streams

| Stream | type | Format | Producer |
|---|---|---|---|
| `UnicornEEG` | `EEG` | 8 ch (`Fz C3 Cz C4 Pz PO7 Oz PO8`, `pipeline.constants.EEG_CHANNELS`), 250 Hz, float32, **µV** | Unicorn LSL Interface on Windows, or `collection/unicorn_relay.py` on a Mac (`collection/README_ACQUISITION.md`); `collection/mock_headset.py` (`MockUnicornEEG`) and `tools/fake_unicorn_bt.py` through the relay (`MockUnicornEEG`) when the data is synthetic |
| `UnicornAux` | `Aux` | 8 ch float64: battery %, accelerometer x y z (g), gyroscope x y z (deg/s), device sample counter; 250 Hz, same timestamps as the EEG | `collection/unicorn_relay.py` only (`MockUnicornAux` when synthetic). Kept apart so the EEG stream stays 8 channels |
| `P300Markers` | `Markers` | 1 ch string, irregular, one JSON event per sample | `checkerboard_stimulus.py` (stamped at `win.flip()`), or the in-window board `collection/board_qt.py` (stamped at the repaint request; see Stimulus boards) |
| `P300Inference` | `P300Inference` | 1 ch string, irregular, one JSON event per sample | `pipeline/inference_service.py` |

`P300Inference` is deliberately **not** type `Markers`. The recorder and the service find the stimulus stream by type `Markers`, and would latch onto the wrong stream if the two shared a type.

Every tool that reads or publishes these streams takes `--eeg-type` / `--marker-type` / `--inference-type` flags. Tests and QA runs use randomized types, so parallel runs never cross-connect.

### Late subscribers

LSL does not replay samples to a consumer that connects after they were pushed.

- **Inlets:** every consumer calls `open_stream()` as soon as it creates an inlet. An unopened inlet buffers nothing; measured here, it received 0 of 10 markers pushed before its first pull.
- **One-shot events can be missed.** The service re-emits `service_status` `ready` every 2 s as a heartbeat. The GUI accepts any `ready` as readiness, and treats the outlet's `desc()` `manifest_fingerprint` as authoritative.
- **The stimulus waits for its consumers.**
  - Before the first trial it blocks until a marker consumer subscribes (`StreamOutlet.wait_for_consumers`, 15 s), then waits 2 s more so the others can subscribe too. Without this, the first `trial_start` of every run was lost.
  - If no consumer appears, it warns and starts anyway. The recorder then flags any trial without a `trial_start`, and the service starts that trial implicitly with a warning.

### Timestamps

EEG and marker inlets in the recorder and the inference service use LSL clock synchronisation only (`proc_clocksync`), **not** dejitter. The fit script inherits this through the recorder's CSV.

Timestamps are recorded as delivered, so the recorder's integrity checks see what the device actually does. Epoching then rejects windows with non-monotonic or gapped timestamps rather than repairing them.

Measured with pylsl 1.18.2, samples matched by ID:
- **Well-behaved outlet:** dejitter changed nothing (0.0 ms) and kept a 3 s dropout gap intact.
- **Outlet that stamps chunks on arrival:** raw timestamps had 7 non-monotonic steps in ~2 s. Dejitter removed them by moving timestamps a median 4.7 ms and at most 18.7 ms, which hides exactly the evidence the recorder should flag.

If the real Unicorn LSL Interface turns out to deliver non-monotonic timestamps (a hardware check in `TESTING.md`), set `EEG_INLET_DEJITTER = True` in `pipeline/constants.py`. The recorder and the service both read that one constant, so training and serving stay consistent.

`collection/mock_headset.py` emits samples on a fixed grid `t0 + index / sfreq`. Its timestamps are strictly increasing, and it leaves a real gap after a stall.

### Timestamps from the relay

`collection/unicorn_relay.py` stamps each sample with `counter / 250 + offset`, where `offset` maps the headset's sample counter onto the LSL clock (`unicorn_protocol.CounterClock`). Bluetooth only adds delay, so `arrival - counter / 250` is at least the true offset and its minimum over a 15 s window estimates it; the offset is slew-limited to 200 ppm, so it can follow the headset's crystal drifting against the host clock while stamps stay strictly increasing. Effects:

- Radio bursts do not show in the stamps (simulated: about 0.1 ms error against about 5 ms when stamping on arrival).
- A dropped packet is a jump in the stamps of `(missing + 1) * 4 ms`, so the recorder's gap detection and epoch rejection see it exactly as before.
- The constant part of the link delay remains, identical in calibration and live use. Only a photodiode measures it (TESTING.md H7, H14).
- The channel labels come from `EEG_CHANNELS`, not from the headset, so the service's label check is vacuous for this producer.

### Why LSL for inference events, not in-process IPC

The service and the GUI are separate OS processes. The GUI launches the service as a subprocess, just as it launches the recorder and the stimulus. LSL is already the project's transport, and it puts inference events on the same clock as EEG and markers, so a recording and its inference log can be aligned afterwards.

## `P300Inference` events

| event | fields |
|---|---|
| `posterior_update` | `trial_id`, `flash_idx`, `max_prob`, `leading_char_idx`, `theta_current` |
| `character_committed` | `trial_id`, `char_idx`, `char`, `n_flashes_used`, `reason` |
| `flash_rejected` *(extension)* | `trial_id`, `flash_idx`, `reason` ∈ {`quality_gate`, `epoch_incomplete`, `classifier_invalid_score`}, `channels` |
| `service_status` *(extension)* | `state` ∈ {`starting`, `ready`, `warning`, `error`, `stopped`}, `detail`, `manifest_fingerprint` |

`reason` for a commit:

- `threshold_crossed`: the posterior max reached `theta` after at least `t_min` flashes.
- `max_flashes_reached`: forced at `t_max`.
- `trial_ended` *(extension)*: the stimulus ended the trial before either condition fired. With the defaults (5 repetitions × 24 flashes = 120 < `t_max` = 170), this is the only fallback that can actually occur. Without its own reason it would pass for one of the other two.

If the controller reports a stop that meets neither condition, the service emits `service_status` `error` and commits nothing. It never relabels such a stop as a fallback.

The GUI shows the two fallback reasons (`events.is_fallback()`) visually distinct from `threshold_crossed`, and never by colour alone.

Validation is strict: unknown event types, missing keys, and extra keys are all errors. A consumer catches `EventSchemaError`, counts it, and surfaces it. One malformed event never crashes the GUI.

## `session_manifest.json`

The schema and every guarantee are in the docstring of `pipeline/manifest.py`. In short:

- It names the characters file, the classifier checkpoint, the Platt calibration file, the language model (`uniform`, `ngram` plus a corpus, or `transformer` plus a checkpoint), the stopping parameters, and the quality-gate policy.
- Artifact sha256s are re-verified at load, and the calibration file must name the sha256 of the classifier it was fitted on.
- The fingerprint is the sha256 of the canonical JSON, so equal fingerprints mean byte-identical models.
- Nothing hardcodes a model path.
  - Defaults: `<repo>/session_manifest.json`, or `<repo>/demo_session_manifest.json` in demo mode.
  - Demo mode never loads a real manifest by default, and real mode never loads a demo one.

## Artifacts written by `pipeline/fit_from_recording.py`

**Classifier checkpoint** (`models/<id>_classifier.joblib`) is a joblib-pickled dict:

```python
{
    "format": "p300-classifier-v1",
    "pipeline": <fitted Pipeline from classifiers/xdawn_cov_ts_regLDA.build_pipeline(), unchanged>,
    "epoch_spec": EpochSpec.to_dict(),
    "labels": {"target": 1, "non_target": 0},
    "score_method": "decision_function",   # positive = target
    "trained_on": {"recording_sha256": ..., "n_epochs": ..., "n_targets": ..., "n_trials": ...},
    "versions": {"numpy": ..., "scipy": ..., "sklearn": ..., "pyriemann": ...},
}
```

joblib is pickle, so loading one runs code. Only load checkpoints you created. The manifest's sha256 check stops a file being swapped behind an unchanged path; it does not make an untrusted file safe.

**Calibration** (`models/<id>_platt.json`):

```json
{"format": "p300-platt-v1", "a": 1.37, "b": -0.79, "score_method": "decision_function",
 "classifier_sha256": "<sha256 of the checkpoint above>", "ece": 0.04, "n_scores": 1200, "n_targets": 100}
```

Platt scaling is fitted on **out-of-fold** decision scores (grouped by `trial_id`), never on scores from a model that saw the same epochs. In-sample scores are overconfident and would inflate every log-likelihood ratio.

**Demo safety:** a recording named `demo_*` is refused unless `--allow-demo` is passed. With it, the checkpoint, calibration, and default manifest are all `demo_`-prefixed. Existing artifacts are never overwritten without `--force`.

## Experiment logs

Every session leaves a record of what was spelled, how the classifier was trained, and how each character was decoded. All files hang off the recording CSV's path (the recorder's own sidecar convention); `pipeline/experiment_log.py` defines the names and schemas, and nothing else spells them.

| File | Written by | Contains |
|---|---|---|
| `<csv>.session.json` | the panel (`gui_logic.SessionLogger`) | **Calibration characters:** `targets_text` (exactly what was typed) and `target_sequence` (one entry per trial: order, character, matrix index). Also `kind`, settings, manifest path and fingerprint, start and end time, how far the session got (`end.phase_at_end`), and the paths of the other files. |
| `<csv>.log.jsonl`, `<csv>.integrity.json` | the recorder | **Calibration process:** connection, gaps, warnings, and the exit summary (see the recorder docstring). |
| `models/<id>_fit_report.json` | `pipeline/fit_from_recording.py` | **Classifier training:** recording sha256 and counts, the embedded `<csv>.session.json` (the calibration sentence), epochs used and rejected, per-fold and out-of-fold AUC, Platt `a`/`b`/ECE, artifact hashes, manifest fingerprint, LM config, library versions, and every console line. |
| `<csv>.inference.log.jsonl` | the inference service (`--log-file`, set by the panel) | **Inference, per character** (records below). |

`kind` is derived: a session that ran with live inference is `inference`, one that only recorded is `calibration`. A recording made before session records existed, or moved away from its sidecars, has no session file; the fit report then says `"calibration_session": null`.

Inference log records, one JSON object per line, keyed by `event` (the `time`, `level`, `logger`, `message` keys are the formatter's). Every 72-long array is aligned to `service_config.characters`, and probabilities are rounded to 6 significant digits.

| `event` | when | fields |
|---|---|---|
| `service_config` | once, at startup | full manifest (every artifact sha256), `characters`, Platt `a`/`b`, `epoch_spec`, EEG stream name, channel labels and rate |
| `trial_start` | each character, before its first flash | `trial_id`, `context` (text committed so far), **`lm_prior`** (the language model's distribution as produced), `initial_posterior` (what the controller starts from: the prior blended with the uniform floor), `lm_prior_top5`, and, when the stimulus cued a target, `target_char_idx`, `target_char`, `target_prior_prob`, `target_prior_rank` |
| `flash_update` | each accepted flash | `trial_id`, `flash_idx`, `t`, `group`, classifier `score`, `pi`, `max_prob`, `leading_char`, `target_prob` |
| `flash_rejected` | each rejected flash | `trial_id`, `flash_idx`, `reason`, `channels` |
| `char_committed` | each decision | `char`, `reason`, `n_flashes_used`, `final_posterior`, `committed_text`, `correct` |

The target fields are **log-only**: the service reads `target_char_idx` from the `trial_start` marker so a commit can be scored (`correct`), and decoding never uses it. They are `null` when the marker carries none.

`lm_prior` is the model's output, not the controller's starting point: with the default `lam = 0.7`, a 99% LM prior starts the posterior at about 0.70. `initial_posterior` is logged next to it so the effect of the blend is visible.

## Channel labels

The service compares the live stream's channel labels with the checkpoint's `EpochSpec.channels`:

- Labels that name electrodes but differ from the checkpoint are a montage mismatch: exit 2.
- Labels that name none of the checkpoint's electrodes (e.g. `EEG 1`…`EEG 8`) produce a warning, and the check falls back to channel count and sample rate.

The real Unicorn LSL Interface's labels are unverified (see `TESTING.md`).

## Recorder CSV

The columns: `timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, marker_json_or_empty, trial_id, flash_idx, group_json, is_target, target_char_idx` (the 8 channel columns are `pipeline.constants.EEG_CHANNELS`).

Besides one row per EEG sample, some rows have all 8 channels NaN:
- gap rows (`stream_gap_start` / `stream_gap_end`);
- stimulus markers that could not be attached to an EEG sample (for example, inside a gap). These carry the marker's own timestamp.

EEG consumers take samples only from rows whose channels are all finite. Side files are `<csv>.log.jsonl` (structured log) and `<csv>.integrity.json` (an end-of-session summary).

## Epochs

Defined by `EpochSpec`, which is stored in the checkpoint:

- Window 0–0.8 s after flash onset, with 0.5 s of padding on each side.
- Zero-phase 1–24 Hz Butterworth band-pass on the padded window only, then cropped; no baseline by default.
- Units: µV throughout.

The service refuses to start if the checkpoint's channels or sample rate don't match the live stream (see Channel labels).

**When a flash gets no epoch** (`flash_rejected` `epoch_incomplete`):
- the window spans a timestamp gap, non-monotonic timestamps, or non-finite samples;
- the EEG stays silent, and the flash is still queued 5 s after its window should have completed. This keeps `trial_end` handling from blocking forever.

Flashes are 124 ms apart and share EEG, so a posterior update lags its flash by roughly `tmax` plus padding.

## Quality gate

The policy is `pipeline/quality.gate_epoch()`; the default is `skip`.

- **`skip`:** a flash is dropped if **any** channel is red or unknown over the epoch window, because xDAWN mixes all channels and one railed electrode contaminates every component.
- **Dropped flashes** never reach the controller, so they count toward neither `t_min` nor `t_max`.
- **Yellow** never drops a flash.
- **`warn`:** accepts everything and emits one `service_status` warning per trial naming the bad channels.

## Stimulus boards

The panel's **Board** setting chooses what draws the flashes. Both publish the same `P300Markers` JSON with the same trial structure (`collection/board_schedule.py` mirrors `checkerboard_stimulus.run_trial()`), and both wait for marker consumers before the first trial.

| Board | Marker timestamp | Use for |
|---|---|---|
| In this window (default) | `local_clock()` right after the repaint request. Measured on this Mac: flash-to-flash 123.97 ms median for a planned 124 ms (sd 1.1 ms), stamps within 7 ms of plan. Lag to the screen unmeasured, expected up to 1–2 refreshes | demos, experimenting |
| Separate window (precise timing) | PsychoPy's `win.flip()` time | real data, until the in-window board passes the photodiode check (TESTING.md H7) |

Qt on macOS offers no presentation timestamp. Neither `QOpenGLWidget.frameSwapped` nor timer-driven repaints are locked to the display refresh (measured median intervals 4–4.5 ms on a 120 Hz screen), and `repaint()` usually returns before the paint happens (0.02 ms median).

## Session start order

`collection/gui_logic.SessionOrchestrator`, driven by the Qt panel's 100 ms refresh:

1. **Inference service** (only when Live inference is ticked). Wait for a `ready` whose fingerprint matches the panel's manifest; 20 s timeout.
2. **Recorder.** Wait until `<csv>.log.jsonl` contains an `"event": "recording_started"` record, meaning EEG is being written; 35 s backstop. The recorder itself exits 3 after 30 s without EEG.
3. **Stimulus board**: drawn in the panel, or the PsychoPy process (see Stimulus boards).

Waiting for `recording_started` guarantees that no EEG is missing when flashing starts. The stimulus then waits for its marker consumers before the first trial, so the first `trial_start` isn't lost either (see Late subscribers).

A child that exits during start-up aborts the session, and its exit code is explained to the operator. When the stimulus exits, the recorder and service are stopped: terminate, then kill after 3 s.

## Exit codes

| Process | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `collection/record_session.py` | normal stop (Ctrl+C, SIGTERM, `--duration`) | unexpected error | bad arguments / output exists | EEG stream not found | — |
| `pipeline/inference_service.py` | stopped (SIGTERM/SIGINT) | crash (traceback in log) | configuration, manifest, checkpoint, or stream mismatch | EEG stream not found | stopping controller method not implemented |
| `pipeline/fit_from_recording.py` | success | — | refused or failed | — | — |

Each non-zero exit prints one explanatory line to stderr. For the service, that line starts `inference_service: exit N:`.

## Known limitation: the stimulus is open-loop

`checkerboard_stimulus.py` runs a fixed number of repetitions and is out of scope this round. When stopping fires, the service commits the character and ignores that trial's remaining flashes. The GUI shows the dynamic-stopping decision, but the subject still sits through the full trial, so ITR gains aren't realised live until the stimulus accepts a stop signal.
