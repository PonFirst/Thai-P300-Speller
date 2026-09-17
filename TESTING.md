# Testing

This document defines "done" for the recorder hardening, live inference service, and PySide6 GUI. Each required failure scenario has an intended behaviour, automated tests that check it, and hardware checks for what can't be automated. A scenario missing from here isn't done, whatever code exists.

## Running the tests

On this machine, a pipenv virtualenv earlier on `PATH` shadows `python` even after `conda activate p300speller`. Call the environment's interpreter by path:

```bash
PY="$(conda info --base)/envs/p300speller/bin/python"
QT_QPA_PLATFORM=offscreen "$PY" -m pytest tests/ -q          # full suite, about 1-2 minutes
```

**Expected: everything passes.** `reset()`, `llr()`, and `update()` in `stopping/bayesian_stopping.py` are now filled in by hand and verified against a hand-worked toy example in `tests/test_bayesian_stopping.py`. Three tests self-skip with "BayesianStoppingController.reset() is implemented now" (`test_fault_injection.py`, `test_inference_service.py` x2) -- they exist to check stub-only behaviour and correctly stand down now that it's filled in.

Per component:

```bash
"$PY" -m pytest tests/test_quality.py tests/test_events.py tests/test_manifest.py tests/test_epoching.py   # contracts
"$PY" -m pytest tests/test_record_session.py                                                              # recorder
"$PY" -m pytest tests/test_classifier_io.py tests/test_language_model.py tests/test_fit_from_recording.py tests/test_inference_service.py
"$PY" -m pytest tests/test_bayesian_stopping.py tests/test_calibrate.py tests/test_lm_prior.py tests/test_evaluate_recording.py  # stopping + replay/ITR
QT_QPA_PLATFORM=offscreen "$PY" -m pytest tests/test_gui_logic.py tests/test_gui_qt_smoke.py              # GUI
"$PY" -m pytest tests/test_stimulus_consumer_wait.py                                                      # stimulus
QT_QPA_PLATFORM=offscreen "$PY" -m pytest tests/test_board_schedule.py tests/test_board_qt.py               # in-window board
QT_QPA_PLATFORM=offscreen "$PY" -m pytest tests/test_fault_injection.py                                   # cross-component
```

- **Real LSL:** some tests start real LSL outlets and subprocesses on loopback. They always use randomized stream types, never `EEG` / `Markers` / `P300Inference`, so they can't cross-connect with each other or with a live session.
- **Clean-up:** no test writes into the repo. `test_fit_tests_never_write_into_the_repo` enforces this for the fit script.

## Required failure scenarios

### 1. An LSL stream never appears

**Intended behaviour**
- **Recorder:** retries with exponential backoff, logging each attempt. With no EEG after `--eeg-timeout` (default 30 s), it exits **3** with one clear stderr line. A missing marker stream only warns, repeatedly, and EEG keeps recording.
- **Inference service:** exits **3** after `--eeg-timeout`, with an `inference_service: exit 3: …` line.
- **Stimulus:** if no marker consumer subscribes within 15 s, it warns and starts anyway.
- **GUI:** shows red "No EEG stream — start Unicorn LSL Interface" and Start is disabled. When a child exits 3, the GUI explains it in plain words. Nothing hangs.

**Automated**
- `test_record_session.py::test_eeg_never_found_exits_3_within_timeout_with_growing_backoff`
- `test_record_session.py::test_marker_timeout_warns_repeatedly_but_recording_continues`
- `test_record_session.py::test_eeg_recorded_before_marker_stream_appears`
- `test_inference_service.py::test_runner_exits_3_without_eeg`
- `test_stimulus_consumer_wait.py::test_no_consumer_warns_and_starts_anyway`
- `test_fault_injection.py::test_recorder_without_eeg_stream_exits_3_and_panel_says_why` (real recorder process → GUI message)
- `test_fault_injection.py::test_service_without_eeg_stream_exits_3_and_panel_says_why` (real service process → GUI message)
- `test_gui_logic.py::test_connection_searching_blocks_start`, `::test_recorder_exit_codes`, `::test_service_exit_codes`
- `test_gui_qt_smoke.py::test_start_disabled_while_searching_or_lost`

**Hardware:** H1.

### 2. A stream drops mid-session

**Intended behaviour**
- **Recorder:**
  - After 0.5 s of silence it writes a `stream_gap_start` row (all channels NaN) straight to disk, so a crash mid-outage still leaves the gap visible.
  - On resume it writes `stream_gap_end`, with the duration and an estimate of the samples missed.
  - It keeps running and never drops a marker: a marker that falls inside the gap is written as its own row.
- **Fit script:** takes EEG only from rows with finite channels, and rejects any epoch whose window spans the gap.
- **Service:**
  - rejects gapped epochs (`flash_rejected` `epoch_incomplete`);
  - after 2 s of silence, warns and pauses the `ready` heartbeat;
  - expires queued flashes 5 s after their window should have completed, so `trial_end` is never blocked.
- **GUI:** within 1 s shows "EEG stream lost", turns every dot gray and blocks Start. It recovers by itself when samples return.

**Automated**
- `test_record_session.py::test_silence_dropout_writes_gap_rows_and_recording_continues`
- `test_record_session.py::test_timestamp_jump_without_stall_is_a_gap`
- `test_record_session.py::test_real_lsl_outlet_restart_records_gap_and_resumes` (real outlet killed and restarted)
- `test_fault_injection.py::test_real_recorder_dropout_is_honoured_by_the_fit_loader` (the real recorder's gap rows → the fit loader; no markers lost during the outage)
- `test_fit_from_recording.py::test_gap_rows_are_not_eeg_and_rejections_are_counted`
- `test_epoching.py::test_gap_inside_window_rejected`, `::test_window_covered_in_time_but_missing_samples_is_a_gap`
- `test_inference_service.py::test_incomplete_epochs_are_rejected_not_scored`, `::test_stale_flash_is_rejected_so_trial_end_is_not_blocked`, `::test_stale_flash_with_no_eeg_at_all`, `::test_eeg_silence_monitor_warns_once_and_reports_recovery`, `::test_runner_heartbeat_generic_labels_and_eeg_silence`
- `test_gui_logic.py::test_connection_connected_then_lost_then_recovered`, `::test_connection_resolved_but_silent_stream_goes_lost`, `::test_scalp_levels_unknown_when_not_delivering_or_empty`

**Hardware:** H5.

### 3. A channel goes flat or railed mid-session

**Intended behaviour**

Thresholds are unchanged: green 1–200 µV peak-to-peak, yellow 200–400 µV, red outside that range.
- **Railed data** turns the dot red at the next 100 ms refresh after it arrives.
- **Flat data** turns it red at the next refresh after **100 ms of flatness**. A channel is red when any 100 ms segment of the window is below 1 µV.
  - Why 100 ms: with whole-window peak-to-peak alone, a contact that failed mid-window stayed green for up to a second.
  - Why not instant: 100 ms is the shortest span that separates a dead contact from a quiet stretch of real EEG.
- **Quality gate (`skip`, the default):** drops any flash with a red or unknown channel. It emits `flash_rejected` `quality_gate` naming the channels, and the flash never reaches the controller.
- **Quality gate (`warn`):** accepts the flash and warns once per trial.
- **Yellow** never drops a flash.
- The GUI map and the gate call the same function, so they can't disagree.

**Automated**
- `test_fault_injection.py::test_mock_fault_colours_scalp_map_and_gate_makes_the_same_call[flat|railed|noisy]`
- `test_fault_injection.py::test_railed_contact_mid_session_shows_red_within_one_refresh`
- `test_fault_injection.py::test_flat_contact_mid_session_shows_red_within_one_refresh_of_100_ms_flatness`
- `test_quality.py::test_threshold_boundaries`, `::test_matches_tkinter_panel_for_every_finite_value`, `::test_flat_onset_mid_window_is_red_after_100_ms_of_flatness`, `::test_railed_onset_is_red_as_soon_as_it_arrives`, `::test_realistic_noise_never_reads_flat`, `::test_gate_rejects_epoch_whose_channel_goes_flat_near_its_end`, `::test_single_red_channel_rejects_whole_flash`, `::test_yellow_channel_never_rejects`
- `test_inference_service.py::test_quality_gate_flat_channel`, `::test_clean_epochs_pass_the_gate_across_seeds`
- `test_gui_qt_smoke.py::test_flat_channel_dot_turns_red_after_one_refresh`
- `test_mock_headset.py::test_bad_channel_simulation_lands_in_intended_quality_bucket`

**Hardware:** H6. **Manual:** GUI checklist item 5.

### 4. The classifier returns NaN or an out-of-range score

**Intended behaviour**
- **Non-finite score:** a NaN or ±inf score becomes `flash_rejected` `classifier_invalid_score`. `update()` is never called, so the posterior is untouched; `np.clip` doesn't remove NaN, so one NaN reaching `llr()` would poison the whole character.
- **Large finite score:** it is passed through, because clipping is the controller's job (`llr_clip`).
- **Invalid posterior:** after any controller call, a non-finite posterior or one that doesn't sum to 1 gives `service_status` `error`, and that trial commits nothing.

**Automated**
- `test_inference_service.py::test_non_finite_scores_never_reach_update`
- `test_inference_service.py::test_invalid_posterior_stops_the_trial_without_commit`
- `test_inference_service.py::test_invalid_language_model_prior_fails_loudly`
- `test_classifier_io.py::test_score_epoch_checks_shape`
- `test_events.py::test_invalid_events_rejected` (NaN or out-of-range `max_prob` can never be published)

**Hardware:** none needed.

### 5. `t_max` fires with no threshold crossing

**Intended behaviour**
- **Forced stop:** the commit reason is `max_flashes_reached`, distinct from `threshold_crossed`.
- **Trial ended first:** if the stimulus ends the trial before either condition fires, the reason is `trial_ended`. With 5 repetitions (120 flashes) and `t_max` = 170, this is the fallback that actually occurs.
- **Neither condition:** a stop that meets neither condition is an error and commits nothing; it is never relabelled.
- **In the GUI:** fallback commits are orange, underlined, and marked ▲ in the composed text, with "▲ FALLBACK" log rows and a tooltip giving the reason and flash count. They are never distinguished by colour alone.

**Automated**
- `test_inference_service.py::test_full_session_event_order_and_commit_reasons`
- `test_inference_service.py::test_stop_without_a_stopping_condition_commits_nothing`
- `test_events.py::test_commit_reasons_are_distinct_and_classified`, `::test_contract_examples_validate`
- `test_gui_logic.py::test_commit_reason_and_fallback_flag`
- `test_gui_qt_smoke.py::test_composed_text_gauge_and_fallback_style`
- `test_fault_injection.py::test_manifest_disagreement_hides_output_while_matching_panel_shows_fallbacks` (scripted publisher → real LSL → GUI model: "ก่อ", with reasons `[threshold_crossed, max_flashes_reached, trial_ended]` and fallback flags `[False, True, True]`)

**Manual:** GUI checklist items 10–11.

### 6. Two processes disagree on the session manifest

**Intended behaviour**
- **Manifest:** every process loads `session_manifest.json`; none hardcodes a model path.
  - Artifact sha256s are re-checked at load, so a checkpoint swapped behind an unchanged path is an error.
  - The calibration file must name the sha256 of the classifier it was fitted on.
- **Service:** publishes the manifest fingerprint in its stream description and in every `service_status`, and re-sends `ready` every 2 s. A late-subscribing GUI can therefore still verify it.
- **GUI:** compares that fingerprint with its own manifest.
  - On a mismatch, or a missing fingerprint, it shows a red banner, hides the output, and never counts the service as ready, so a session with Live inference won't start.

**Automated**
- `test_manifest.py::test_classifier_swapped_behind_same_path_is_detected`, `::test_calibration_fitted_for_another_classifier_is_detected`, `::test_fingerprint_changes_with_any_setting`, `::test_fingerprint_ignores_key_order_and_whitespace`, `::test_assert_same_fingerprint`
- `test_fault_injection.py::test_late_subscribing_panel_verifies_service_manifest_via_heartbeat` (real service process: the matching panel becomes ready through the heartbeat; a panel with a different fingerprint shows a mismatch)
- `test_fault_injection.py::test_manifest_disagreement_hides_output_while_matching_panel_shows_fallbacks` (matching, different, and no-manifest panels on one stream)
- `test_inference_service.py::test_runner_over_real_lsl`
- `test_gui_logic.py::test_desc_fingerprint_mismatch_hides_stream`, `::test_status_fingerprint_mismatch_mid_stream_stops_display`, `::test_desc_is_authoritative_even_when_statuses_match`, `::test_panel_without_manifest_cannot_verify_any_stream`, `::test_ready_heartbeat_alone_is_enough`, `::test_ready_from_a_stream_connected_before_start_does_not_count`
- `test_gui_qt_smoke.py::test_manifest_mismatch_shows_error_and_hides_output`

**Manual:** GUI checklist item 12.

### 7. Demo mode and real mode files collide

**Intended behaviour**
- **Recording names:**
  - Demo recordings are always `demo_`-prefixed.
  - A real session ID may not be `demo` or start with `demo_`.
  - A generated recording path never reuses an existing file.
- **Recorder:** refuses an existing `--out` and exits **2**.
- **Fit script:**
  - refuses a `demo_` recording without `--allow-demo`;
  - with it, prefixes every artifact `demo_`;
  - never overwrites without `--force`.
- **GUI manifest default:** `demo_session_manifest.json` in demo mode and `session_manifest.json` otherwise, both at the repo root.

**Automated**
- `test_fault_injection.py::test_demo_and_real_recordings_can_never_share_a_file` (GUI paths + real recorder refusal for both files + fit script prefixes)
- `test_gui_logic.py::test_recording_path_real_and_demo`, `::test_recording_path_never_returns_an_existing_file`, `::test_demo_path_collision_gets_suffix_and_stays_demo`, `::test_existing_demo_file_does_not_affect_real_path`, `::test_real_session_cannot_produce_a_demo_named_file`, `::test_default_manifest_depends_on_mode_and_ignores_cwd`
- `test_record_session.py::test_existing_output_is_refused_with_exit_2`
- `test_fit_from_recording.py::test_demo_recording_refused_without_flag`, `::test_demo_with_flag_prefixes_everything_and_refuses_overwrite`, `::test_fit_tests_never_write_into_the_repo`
- `test_gui_qt_smoke.py::test_manifest_defaults_to_the_mode_file_at_repo_root`

**Known gap:** a manifest named explicitly with `--manifest` isn't checked against the mode. A demo panel will load a real manifest if you pass one.

## Other guarantees with automated tests

| Guarantee | Tests |
|---|---|
| An unimplemented stopping controller fails loudly, at the exact call, naming the method | `test_inference_service.py::test_update_not_implemented_fails_loudly_at_that_call`, `::test_real_controller_stub_names_reset`, `::test_runner_exits_4_when_reset_is_a_stub`; `test_fault_injection.py::test_unimplemented_controller_stops_service_with_exit_4_naming_the_method`; `test_gui_logic.py::test_service_exit_4_before_ready_aborts_with_method_name`; `test_gui_qt_smoke.py::test_service_not_implemented_exit_is_explained` |
| The stimulus holds its first trial until marker consumers have subscribed | `test_stimulus_consumer_wait.py`; `tools/e2e_demo_smoke.sh` (the integrity summary shows no trials without `trial_start`) |
| The in-window board plays the PsychoPy stimulus's trial structure and marker JSON on schedule, and waits for consumers | `test_board_schedule.py`; `test_board_qt.py` (including a real LSL consumer); `test_gui_logic.py::test_in_window_board_runs_instead_of_a_stimulus_process`; `test_gui_qt_smoke.py::test_in_window_board_replaces_the_traces_and_publishes_markers` |
| The stimulus starts only once the recorder is writing EEG | `test_gui_logic.py::test_calibration_start_without_manifest`, `::test_recorder_that_never_starts_recording_times_out`, `::test_recorder_started_reads_the_recorders_structured_log`; `test_fault_injection.py::test_real_recorder_dropout_is_honoured_by_the_fit_loader` (the GUI's readiness check against the real recorder's log) |
| Markers attach to the right sample and are never silently lost | `test_record_session.py::test_normal_operation_attaches_markers_to_nearest_sample_including_late_one`, `::test_late_and_colliding_markers_written_as_own_rows`, `::test_duplicate_redelivered_and_orphan_flash_markers`, `::test_marker_outlet_replaced_reconnects_and_resets_tracking`, `::test_sigterm_flushes_held_rows_and_writes_summary` |
| Timestamps are recorded as delivered; one dejitter switch for recorder and service | `test_record_session.py::test_non_monotonic_timestamps_flagged_not_reordered_or_dropped`, `::test_inlet_processing_flags_follow_shared_dejitter_switch`; `test_inference_service.py::test_eeg_inlet_flags_follow_the_dejitter_constant`; `test_mock_headset_timing.py` |
| Training and serving cut identical epochs | `test_epoching.py::test_live_buffer_and_full_recording_give_identical_epochs` |
| Checkpoint/stream compatibility and channel-label policy | `test_classifier_io.py::test_incompatible_stream_names_the_field`, `::test_channel_label_policy`; `test_inference_service.py::test_runner_exits_2_on_montage_mismatch`, `::test_runner_heartbeat_generic_labels_and_eeg_silence` |
| Calibration never uses in-sample scores | `test_fit_from_recording.py::test_out_of_fold_scores_never_score_a_trial_the_model_saw` |
| Thai tone marks stack correctly in the GUI font | `test_gui_qt_smoke.py::test_thai_tone_mark_stacks_above_its_base` |

## End-to-end smoke test (no hardware; opens a window)

```bash
tools/e2e_demo_smoke.sh
```

**What it runs:**
1. The mock headset, the real recorder, and the real PsychoPy stimulus (3 characters × 3 repetitions).
2. The fit script, on the CSV that the recorder actually wrote.
3. The inference service, until it reaches `ready` on that model.
4. A P300 check at Pz.

It exits 0 only if every stage succeeds.

**Last run (2026-09-11, after the stimulus consumer wait):**
- **Exit codes:** all stages exited 0.
- **Recording:** 216/216 flashes captured and all epochs usable, with 0 non-monotonic timestamps.
- **Integrity summary:** no issues. Before the wait was added, every run reported "72 flashes without trial_start".
- **Signal:** the Pz target average peaked at +4.42 µV at 384 ms, against the mock's ~6 µV at ~350 ms injection with per-flash jitter, averaged over 18 target epochs.
- **AUC:** out-of-fold AUC was 0.68. That is expected from the mock's weak signal and tiny calibration set; it is not a measure of the classifier.

## Manual QA checklist: GUI

These can't be fully automated because they depend on the Qt event loop and a real display. `$PY` is as above; run from the repo root. (V) means the GUI agent verified the item with screenshots on this Mac; re-check the rest on screen.

**Setup:** create a dummy manifest for the display items.

```bash
$PY - <<'EOF'
import json, pathlib, sys; sys.path.insert(0, ".")
from pipeline import manifest as mf
r = pathlib.Path("/tmp/p300qa"); (r/"models").mkdir(parents=True, exist_ok=True)
c = r/"models/clf.joblib"; c.write_bytes(b"qa-fake")
p = r/"models/platt.json"; p.write_text(json.dumps({"format": mf.CALIBRATION_FORMAT, "a": 1.0, "b": 0.0, "score_method": mf.SCORE_METHOD, "classifier_sha256": mf.sha256_file(c)}))
print(mf.write_manifest(mf.build_manifest_dict(base_dir=r, characters_path="collection/characters_th.json", classifier_path=c, calibration_path=p), r/"session_manifest.json").fingerprint)
EOF
```

1. **Demo launch.** Run `./start_speller.sh --demo`.
   - A yellow "DEMO MODE — synthetic data" banner, and "[DEMO]" in the title.
   - The status goes from red "● No EEG stream — start Unicorn LSL Interface" to green "● EEG connected".
   - The manifest note names `<repo>/demo_session_manifest.json`, or says it is missing with a `--allow-demo` hint. (V)
2. **Real-mode default.** Run `$PY collection/session_gui_qt.py`. The note names `session_manifest.json`, never the demo manifest. (V)
3. **Traces.** Top to bottom: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, over a 0–4 s axis, all visible at a similar height. (V)
4. **Scalp map.** Two lobed hemispheres with a centre gap, 8 labelled dots, and a 4-entry legend (Good / Noisy / Bad-flat / No data). (V)
5. **Fault injection.** Run `$PY collection/mock_headset.py --flat PO7 --noisy C4 --railed Oz`, then `$PY collection/session_gui_qt.py`. PO7 is red, C4 yellow, Oz red, and the rest green. (V for flat and noisy)
6. **Stream lost.** Press Ctrl+C on the mock.
   - Within about 1 s: red "● EEG stream lost", Start disabled, all dots gray.
   - Restart the mock: the status returns to green and Start is enabled. (V)
7. **Defaults and validation.**
   - Defaults are subject_001, 5, 62, 62, 0, Windowed off.
   - Session IDs `demo`, `demo_x` and `a b` are refused with a red message.
   - Target `กA` gives "Not in the 72-character matrix: A". (V)
8. **Calibration run.** Run `./start_speller.sh --demo`, type `กข`, set Repetitions to 1, press Enter.
   - The recorder launches, and `collection/recordings/demo_subject_001_<stamp>.csv` appears.
   - With Board = "In this window" (the default), the board replaces the traces only after the recorder has logged `recording_started`, and waits about 2 s before its first cue.
   - With Board = "Separate window (precise timing)", the PsychoPy window opens instead, and Screen and Windowed apply.
   - When it finishes: "Stimulus finished — stopping recorder.", and the traces come back.
9. **Stop and close.** Click Stop mid-session: "Stopped." Separately, close the window mid-session. Either way, `pgrep -fl "record_session|checkerboard_stimulus|inference_service"` returns nothing within 3 s.
10. **Live display.** Run `./start_speller.sh --demo --manifest /tmp/p300qa/session_manifest.json`, then `$PY tools/fake_inference_publisher.py --manifest /tmp/p300qa/session_manifest.json --include-malformed`.
    - The service shows "ready".
    - The gauge climbs toward the θ line and empties on each commit.
    - The composed text reads "ก่อ" with the tone mark stacked.
    - The counters read "1 quality gate (last: Pz)" and "1 malformed inference event". (V)
11. **Fallback style.** In the same run:
    - ่ (170 flashes) and อ (60 flashes) are orange, underlined, and marked ▲.
    - Their log rows read "▲ FALLBACK".
    - Hovering shows the reason and the flash count. (V, except hover)
12. **Manifest mismatch.** Run `$PY tools/fake_inference_publisher.py --fingerprint 0000`. A red "…session manifest mismatch…" banner appears and no characters are shown. (V)
13. **Service configuration error.** Tick Live inference with the dummy manifest and press Start. The message reads "Live inference refused to start: configuration, manifest or checkpoint error: … Session not started.", with no liblsl log lines in it. The recorder is never launched. (V)
14. **Live controller run.** Use a real manifest from `pipeline.fit_from_recording`, tick Live inference, and press Start.
    - "Live inference ready.", then the recorder and stimulus launch.
    - `reset()`/`llr()`/`update()` are filled in now (see the hard-constraints note in `CLAUDE.md`), so the trial should run to a real commit instead of erroring: the gauge advances flash by flash and a character is composed. This item described the old `NotImplementedError` stub path before that; it needs a fresh hardware re-check against the implemented controller (no (V) yet).
15. **Tk fallback.** With the mock running, `$PY collection/session_gui.py --demo` still opens the old panel.

## Hardware-in-the-loop checks (Unicorn Hybrid Black)

Run these on the deployment Windows machine, with the real headset. None of
these can run on the Mac this repo is developed on, since it never touches
the device. Do them before the first real data collection; each is a
go/no-go.

- **H1. Stream discovery and shape.** Start the Unicorn LSL Interface, then:
  ```bash
  $PY -c "from pylsl import resolve_byprop as r; print([(s.name(), s.type(), s.channel_count(), s.nominal_srate()) for s in r('type','EEG',timeout=5)])"
  ```
  - **Go:** type `EEG`, 8 channels, 250 Hz.
  - **Stop:** a different channel count (e.g. accelerometer, gyroscope, battery or counter channels included) breaks the 8-channel contract. If nothing resolves and you're running the split-machine variant (collection/README_ACQUISITION.md), check the VM's network mode, since LSL discovery uses UDP multicast.
- **H2. Channel labels and montage.** Print the stream's channel labels (`info.desc().child("channels")`) and compare them with `EEG_CHANNELS` = Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, and with the physical electrode positions (CLAUDE.md gotcha 9). This is the one check this list still hasn't had against a live stream -- the montage itself is verified against the vendor manual (CLAUDE.md hard constraints), but the real device's *reported* labels are not.
  - **Generic labels** (`EEG 1`…) make the service warn and fall back to channel count and rate.
  - **Different electrode names** make it refuse (exit 2) until the montage is confirmed.
- **H3. Units.** Run `$PY collection/record_session.py --out /tmp/h3.csv` for 30 s. In `/tmp/h3.csv.integrity.json`, `sanity_check` shows every channel within 1–200 µV peak-to-peak. Values in the millions mean volts, not µV (gotcha 1).
- **H4. Timestamp behaviour.** Record 2 minutes, then read the integrity summary: non-monotonic steps, `max_irregular_step_s`, `effective_rate_hz`.
  - **Go:** 0 non-monotonic steps, and a rate within ±2 % of 250 Hz.
  - **If there are non-monotonic steps:** set `EEG_INLET_DEJITTER = True` in `pipeline/constants.py` (measured safe across dropouts; see docs/INTERFACES.md), then re-run H4, the full test suite, and a fresh calibration fit.
- **H5. Real dropout.** During a 1-minute recording, switch the headset off for about 5 s (also try walking out of Bluetooth range).
  - **Expect:** the GUI shows "EEG stream lost" within 1 s and recovers; the CSV has gap start/end rows; the recorder exits 0 at the end.
  - **Record whether the LSL Interface keeps the same `source_id` after reconnecting.** Automatic recovery depends on it; if it changes, the recorder and service won't reattach on their own.
- **H6. Electrode contact.** Lift one electrode mid-session: its dot turns red within about 200 ms, and with Live inference on, the quality-gate rejection counter rises. A blink turns frontal dots yellow at most, and no flash is rejected for it.
- **H7. Stimulus timing.** Put a photodiode on a flashing cell and record it alongside the markers. Measure the lag between marker timestamps and actual screen onset, and its jitter, at the display's real refresh rate.
  - Do it for both boards. The in-window board stamps a marker when it requests the repaint; on this Mac its flashes stayed within 7 ms of schedule, but its lag to the screen is unmeasured (expected up to about 17 ms at 120 Hz).
  - Use Board = "Separate window (precise timing)" for real data until the in-window board passes.
- **H8. Stimulus Thai rendering.** `THSarabunNew.ttf` isn't in the repo, so the PsychoPy stimulus falls back to a default font. Run `verify_thai_font_rendering()` and check the tone marks by eye. The GUI's Thai rendering is covered by the automated test; the stimulus's is not.
- **H9. First-trial label.** After a real session, `trials_without_trial_start` in the integrity summary should be 0. A non-zero count means no consumer subscribed within the stimulus's 15 s + 2 s wait; check the recorder log for when the marker stream connected.
- **H10. Live inference end to end** (once `reset`/`llr`/`update` are implemented): record a calibration session, then run `pipeline.fit_from_recording`. Then do a live session with Live inference on, and compare the composed text with the cued targets.
- **H11. Sustained load.** Run a 10-minute live session. `posterior_update` should keep lagging each flash by only about 1.3 s, without growing.

## Known limitations (not done this round)

- **In-window board timing.** Its marker timestamps are estimates (see H7); record real data with the separate PsychoPy window until it has been checked.
- **Open-loop stimulus.** It runs a fixed number of repetitions, so dynamic stopping is decided and displayed but the subject still sits through the whole trial.
- **Manifest vs mode.** An explicit `--manifest` isn't checked against demo or real mode.
- **Real-device behaviour is unverified.** Timestamp behaviour, channel labels and reconnect `source_id` (H2, H4, H5) are all unknown until tested on the headset.
- **Mock AUC.** AUC on mock data is low by design; don't read it as classifier quality.
- **GUI input.** Tooltip hover and Thai keyboard input in the GUI aren't automated.
- **Legacy panel.** The Tk panel is kept as a fallback and isn't covered by the new tests.
