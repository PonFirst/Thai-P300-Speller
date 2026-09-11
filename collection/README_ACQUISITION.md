# Acquisition setup: Unicorn Hybrid Black + macOS (Apple Silicon)

## Why this path, not BrainFlow

Confirmed directly from primary sources (not relitigating this):

- BrainFlow's `unicorn_board.h` wraps the entire Unicorn board class in
  `#if defined __linux__ || defined _WIN32` -- it does not compile on
  macOS at all, Intel or Apple Silicon. `pip install brainflow` will not
  give you a working Unicorn path here.
- g.tec's own Unicorn Suite / Unicorn LSL Interface is Windows-only.

So on this Apple Silicon development machine, the headset itself is
never touched by code in this repo. The only two viable acquisition
paths are:

1. **Windows-on-ARM VM** (e.g. Parallels) running the official Unicorn
   Suite, with the Unicorn's USB dongle passed through to the VM.
2. **A separate Linux machine** running the official vendor stack.

Either way, the *actual* Unicorn driver runs somewhere other than this
Mac, and publishes an LSL outlet that this repo's scripts consume over
the network (or loopback, if using a VM on the same physical machine).
`collection/record_session.py` only ever talks LSL -- it has no
BrainFlow dependency and never imports a Unicorn SDK.

## Machine topology

```
+----------------------------------+       LSL (network or VM loopback)
| Windows-on-ARM VM (or Linux box) | -------------------------------+
|  - Unicorn USB dongle attached   |                                |
|  - Unicorn LSL Interface app     |                                |
|    -> publishes 'UnicornEEG'     |                                |
|       LSL outlet (8ch, 250Hz)    |                                v
+----------------------------------+                    +----------------------+
                                                          | macOS (this machine)|
+----------------------------------+   LSL (localhost)   |                      |
| checkerboard_stimulus.py         | -------------------> | record_session.py   |
|  -> publishes 'P300Markers'      |                      | (resolves BOTH      |
|     LSL outlet (JSON strings)    |                      |  outlets, merges,   |
+----------------------------------+                      |  writes one CSV)    |
                                                           +----------------------+
```

## Startup sequence -- order matters

1. **Start the Unicorn LSL Interface** on the Windows VM / Linux box.
   Connect to the headset, click Start streaming. Confirm in that app
   that it is actively streaming before moving on.
2. **Start `collection/record_session.py`** on this Mac. It resolves the
   EEG stream by type (`--eeg-type`, default `EEG`) with exponential
   backoff: retries after 0.5 s, 1 s, 2 s, 4 s, then every 5 s, logging
   each attempt with the elapsed time and the limit. If no EEG stream
   appears within `--eeg-timeout` (default 30 s), it prints one error
   line on stderr and exits with code 3. **EEG recording starts the
   moment the EEG stream is found**; pre-stimulus EEG is kept.
3. **Start `collection/checkerboard_stimulus.py`** (the PsychoPy
   stimulus window). The recorder keeps looking for the marker stream
   (`--marker-type`, default `Markers`) from inside its recording loop,
   with short probes (0.1 s, every 0.25 s), and opens the marker inlet
   the moment it finds it. pylsl buffers nothing for an inlet that has
   not been opened. If no marker stream appears within `--marker-timeout`
   (default 120 s), the recorder prints a loud WARNING, repeats it every
   30 s, and keeps recording EEG.

**Why this order**: EEG must be streaming within 30 s of starting the
recorder, or the recorder gives up (exit 3). The stimulus can start any
time after the recorder. Wait for the recorder's
`Recording EEG; searching for the marker stream.` line (`recording_started`
in the JSON log) before starting it.

**Known remaining marker loss.** The stimulus creates its marker outlet
and pushes the first `trial_start` almost immediately. No LSL consumer
resolves that fast, so the first trial's `trial_start` can still be
lost. The recorder cannot prevent this without a stimulus change. It
does detect it: every flash whose `trial_id` has no preceding
`trial_start` from that outlet is flagged (`flash_without_trial_start`
in the log, and `trials_without_trial_start` in the integrity summary).

**Stimulus restarted mid-recording.** If the new outlet has a different
uid, the recorder reconnects and logs `marker_outlet_replaced`. With the
same `source_id`, pylsl also recovers the existing inlet on its own, but
the first 1-2 markers of the new outlet can be missed. The stimulus
renumbers trials from 0, so a repeated `trial_start`, or a repeated
(`trial_id`, `flash_idx`) seconds later, is treated as a new run
(`marker_run_restart_suspected`). Duplicate tracking then resets instead
of discarding the new run's flashes.

## Running it

On the Windows VM / Linux box: start the Unicorn LSL Interface and click
Start.

On this Mac, the normal path is the session panel. It starts the recorder,
waits until the recorder's log reports `recording_started`, and only then
starts the stimulus:

```bash
./start_speller.sh
```

To run the two by hand, call the environment's interpreter by path. On this
machine a pipenv virtualenv earlier on `PATH` shadows `python` even after
`conda activate`, and would run the wrong packages.

```bash
PY="$(conda info --base)/envs/p300speller/bin/python"
"$PY" collection/record_session.py --out collection/recordings/session_001.csv
```

Then, in a second terminal, once `record_session.py` logs that it is
recording and searching for the marker stream:

```bash
"$PY" collection/checkerboard_stimulus.py
```

Useful flags: `--duration <s>` (stop automatically, counted from when EEG
is found), `--eeg-timeout`, `--marker-timeout`, `--eeg-type` /
`--marker-type` (for isolated tests; never reuse the production types in
tests), and `--log-file`. `--poll-interval` and `--max-wait` still work.
They now set the backoff cap and the EEG timeout.

## Stopping and exit codes

Stop with Ctrl+C, or send SIGTERM (what the session panel's Stop button
does). The two are handled identically. Rows still held in memory are
written, the integrity summary is written, and the process exits with 0.

| Code | Meaning |
|---|---|
| 0 | Normal stop: Ctrl+C, SIGTERM, or `--duration` elapsed |
| 1 | Unexpected error (details and traceback in the JSON log) |
| 2 | Bad arguments or output path problem, including `--out` already existing. The recorder never overwrites a file |
| 3 | EEG stream not found within `--eeg-timeout` |

## Output format

One CSV per session (see `record_session.py`'s module docstring for why
CSV over Parquet), columns:

```
timestamp, Fz, F3, C4, Cz, Pz, P3, O2, O1, marker_json_or_empty,
trial_id, flash_idx, group_json, is_target, target_char_idx
```

- `timestamp`: LSL timestamp, mapped into this Mac's `local_clock()`
  domain (the inlets use `proc_clocksync`). Without that mapping, EEG
  timestamps from the VM would stay in the VM's clock and could not be
  aligned with local markers. Whether EEG also uses `proc_dejitter` is
  one switch shared with the inference service:
  `pipeline/constants.EEG_INLET_DEJITTER` (currently off, so integrity
  checks see timestamps as delivered). The effective flags are recorded
  in the log and the integrity summary.
- `Fz`...`O1`: the 8 Unicorn channels, one row per EEG sample (250 Hz).
- `marker_json_or_empty`: the raw JSON string pushed by
  `checkerboard_stimulus.py` if a marker landed closest to this EEG
  sample's timestamp, else empty. Markers are timestamped at the GPU
  flip and can arrive after the EEG around them. The recorder therefore
  holds the most recent 1 s of rows (`--holdback`) before writing, so a
  late marker still reaches its true nearest sample, even across chunk
  boundaries.
- `trial_id`, `flash_idx`, `group_json`, `is_target`, `target_char_idx`:
  the same marker JSON pre-parsed into columns, so downstream analysis
  never has to re-parse the JSON string. Empty for every non-marker row.
  `group_json` is the full list of character indices in the flashed
  group -- this is the field the original reference implementation never
  logged, and its absence is what made that data useless for replay
  (see docs/CLAUDE.md and stopping/replay.py). Do not regress this.

### Rows with all 8 channels NaN

These rows are not samples. They are always identified by
`marker_json_or_empty`:

- **`{"event": "stream_gap_start", ...}`**: EEG stopped arriving. It is
  written as soon as the stream has been silent for 0.5 s
  (`--silence-threshold`), so a crash mid-outage still leaves the gap
  visible. It is also written when timestamps jump by more than 0.1 s
  (`--gap-jump`) without a stall (`"detected_by": "timestamp_jump"`).
  Parsed marker columns are empty.
- **`{"event": "stream_gap_end", "duration_s": ..., "n_missing_est": ...,
  "last_sample_ts": ..., "first_sample_ts": ...}`**: written when EEG
  resumes. `duration_s` is the span of missing samples, taken from the
  timestamps on either side. The recorder keeps the inlet open through
  an outage; pylsl reconnects on its own when an outlet with the same
  `source_id` returns. Parsed marker columns are empty.
- **A stimulus marker on its own row**: `timestamp` is the marker's own
  LSL timestamp, and the parsed marker columns are filled. This happens
  when the marker could not attach to its nearest sample, for one of
  three reasons: that sample was already written (marker more than
  `--holdback` late); no EEG sample lies within 1.5 sample periods (the
  marker falls inside a gap); or that sample already carries an earlier
  marker. Every case is logged. **No marker is ever dropped** except an
  exact duplicate, as described in the integrity summary below.

Consumers must take EEG only from rows with finite channels, and events
from every row with a non-empty `marker_json_or_empty`. The file is not
guaranteed to be sorted by timestamp. A marker row written late sits after
samples that were already on disk. Non-monotonic timestamps from the
device are flagged but never reordered or dropped.

### Side files

- **`<csv>.log.jsonl`** (or `--log-file`): one JSON object per line with
  `time`, `level`, `event`, `msg`, and event-specific fields. The console
  shows the same messages concisely (warnings on stderr). A session can
  be diagnosed from this file alone. Key events:
  - startup: `recorder_start` (full config), `eeg_resolve_attempt`,
    `eeg_connected`, `eeg_not_found`
  - markers: `marker_connected`, `marker_timeout`, `marker_outlet_replaced`,
    `marker_outlet_not_visible`, `marker_run_restart_suspected`
  - signal: `sanity_channel` / `sanity_check`, `stream_gap_start` /
    `stream_gap_ongoing` / `stream_gap_end` / `stream_gap_timestamp_jump`,
    `eeg_rate_drift`
  - integrity: `eeg_timestamp_non_monotonic`, `duplicate_flash_marker`,
    `flash_without_trial_start`, `marker_standalone_row`
  - exit: `integrity_summary`, `shutdown` (reason and rows written)
- **`<csv>.integrity.json`**: written at exit, including after SIGTERM:
  - `shutdown_reason`, `exit_code`
  - `issues`: a human-readable list, empty for a clean session
  - `rows_written` (EEG / gap / standalone marker)
  - `sanity_check`, `inlet_processing_flags`, `marker_connections`
  - `gaps` (durations and missing-sample estimates)
  - `effective_rate_hz` (min/max over rolling 10 s windows)
  - `trials_without_trial_start`, `standalone_marker_rows_by_reason`
  - `counts`:
    - non-monotonic EEG timestamps, split into duplicate and backwards
    - irregular timestamp steps: above 1.5 periods but below the gap
      threshold, counted only
    - rate-drift warnings: beyond ±2 % of nominal
    - duplicate flash markers: same outlet, `trial_id` and `flash_idx`
      within 1 s; only the first is kept
    - redelivered markers: identical payload and timestamp, dropped
    - flashes without `trial_start`
    - marker reconnections

## Signal sanity check (first 2 s of recording)

This check no longer blocks before recording. The recorder evaluates
each channel's peak-to-peak amplitude over the first 2 s of samples as
they are recorded (500 samples at 250 Hz), against the same
plausibility band as before (1-200 uV). It warns LOUDLY (`*** WARNING
***` lines on stderr, plus a `sanity_check` record in the log and
summary) if a channel looks flat/disconnected or clipped/railed/wrong
units. It does not abort the session. Investigate any warning before
trusting the data. If the session ends before 2 s of EEG arrives, the
check runs on what was received and is marked `partial`.

## Troubleshooting

- **`record_session.py` logs "Waiting for EEG stream" and exits with
  code 3 after `--eeg-timeout`** -- the Unicorn LSL Interface on the VM/Linux box
  either isn't running, isn't actually streaming (check it says
  "Streaming" in its own UI, not just "Connected"), or is unreachable
  over the network (VM network mode, firewall). LSL discovery uses UDP
  multicast by default -- if the VM and Mac are on different virtual
  network segments, that mode of discovery can silently fail even
  though the VM is otherwise reachable.
- **Sanity check reports all channels flat (~0 uV)** -- electrodes not
  making contact, or the Unicorn isn't actually powered on despite the
  LSL Interface showing "Streaming."
- **Sanity check reports amplitudes in the thousands/millions** -- almost
  certainly a units bug somewhere in the chain (see docs/CLAUDE.md gotcha
  #1 for the exact failure mode this project has hit before with a
  similar units mismatch), not real signal.
