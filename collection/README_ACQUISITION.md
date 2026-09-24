# Acquisition setup: Unicorn Hybrid Black

## Why not BrainFlow

Confirmed directly from primary sources (not relitigating this):

- BrainFlow's `unicorn_board.h` wraps the entire Unicorn board class in
  `#if defined __linux__ || defined _WIN32` -- it does not compile on
  macOS at all, Intel or Apple Silicon. `pip install brainflow` will not
  give you a working Unicorn path on a Mac.
- g.tec's own Unicorn Suite / Unicorn LSL Interface is Windows-only.

So the recorder, the inference service, the panel and the stimulus never
talk to the headset: they are pure LSL consumers, and
`collection/record_session.py` has no BrainFlow dependency and never
imports a Unicorn SDK. Something has to turn the headset's Bluetooth
stream into LSL: on Windows that is g.tec's Unicorn LSL Interface, on a
Mac it is this repo's own `collection/unicorn_relay.py` (see "Mac" below).

## Machine topology: one Windows machine (primary, documented path)

The headset driver, the Unicorn LSL Interface, and every process in
this repo (`record_session.py`, `pipeline/inference_service.py`,
`collection/session_gui_qt.py`, and the stimulus) run together on one
Windows machine the project owner fully controls. Every LSL stream is
still resolved **by type**, never by hostname, so nothing here cares
that everything happens to be on one box:

```
+--------------------------------------------------------------+
| One Windows machine                                          |
|                                                                |
|  Unicorn headset (USB) -> Unicorn LSL Interface               |
|                             -> publishes 'UnicornEEG'          |
|                                LSL outlet (8ch, 250Hz)          |
|                                      |                          |
|                                      v  LSL (localhost)         |
|  checkerboard_stimulus.py  -----> record_session.py            |
|   -> publishes 'P300Markers'        (resolves BOTH outlets,    |
|      LSL outlet (JSON strings)       merges, writes one CSV)   |
|                                                                |
|  collection/session_gui_qt.py drives all of the above          |
|  pipeline/inference_service.py (optional, live inference)      |
+--------------------------------------------------------------+
```

This development repo itself is edited on a separate Apple Silicon
Mac (BrainFlow/Unicorn Suite don't run there at all -- see above), but
that machine is not part of the acquisition topology; it never runs a
real session.

### Split-machine variant (still works, no longer primary)

An earlier setup ran the headset driver on a separate Windows-on-ARM
VM or Linux box, with the rest of the pipeline on another machine,
connected over the network instead of localhost. Nothing in this repo
assumes one machine over the other -- LSL discovery works the same way
either way -- so that split still works if ever needed. The one
practical difference: LSL discovery defaults to UDP multicast, which
can silently fail across virtual network segments; on one machine that
concern doesn't arise.

## Mac: `collection/unicorn_relay.py` (no g.tec software)

The relay does what the Unicorn LSL Interface does and nothing else. It
reads the headset's Bluetooth serial stream, decodes g.tec's documented
45-byte payloads (UnicornBluetoothProtocol.pdf in
github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black), and publishes
`UnicornEEG` (type `EEG`, 8 ch, 250 Hz, float32, microvolts) plus a
separate `UnicornAux` stream (battery, accelerometer, gyroscope, device
sample counter). Every consumer resolves by type, so nothing downstream
changes.

**Status: tested against a protocol-accurate fake headset, a
pseudo-terminal and the real recorder** (`tests/test_unicorn_protocol.py`,
`tests/test_unicorn_relay.py`); **not yet run against a real Unicorn.** That
is TESTING.md H12-H14, and H12 comes first.

1. **Pair the headset** in System Settings > Bluetooth (switch it on; it
   shows as `UN-<serial>`). macOS creates `/dev/cu.UN-<serial>`;
   `python collection/unicorn_relay.py --list-ports` lists it.
2. **Start the relay:** `python collection/unicorn_relay.py` (or
   `--port /dev/cu.UN-<serial>`). It waits up to 30 s for the port, starts
   acquisition, and prints `publishing 'UnicornEEG' ...` about 2 s after the
   first packet (the clock warm-up). Ctrl+C sends the stop command and
   writes a report.
3. **Then the usual sequence** below (recorder, stimulus, `./start_speller.sh`).
   Stop the relay last.

**Run the sample-counter gap test before trusting a session** (H12):
`python collection/unicorn_relay.py --duration 1800 --report logs/gap_test.json`.
The device numbers every sample, so `samples.missing` and `loss_pct` in the
report are true losses, whatever the timing of delivery. The built-in
Bluetooth shares a chip and antenna with Wi-Fi; if it drops packets, that
is the first thing to change (a USB dongle needs a user-space Bluetooth
stack such as Google's Bumble, whose `rfcomm_bridge` exposes the link as a
TCP socket the relay reads with `--tcp HOST:PORT`; not tested here).

What is different from the Windows path, and worth knowing:

- **Channel labels are written by the relay**, from `EEG_CHANNELS`. The
  headset sends none, so the service's electrode-label check cannot catch
  a wrong cap here. Check physically that channels 1-8 are Fz C3 Cz C4 Pz
  PO7 Oz PO8 (H13).
- **Timestamps come from the headset's sample counter**, not from when
  Bluetooth delivered the bytes (`unicorn_protocol.CounterClock`). Radio
  bursts therefore do not smear the ERP: in simulation the stamp error is
  about 0.1 ms, against about 5 ms when stamping on arrival. A lost packet
  is a jump in the stamps, which the recorder's gap detection already
  handles. The constant part of the link delay is not measurable in
  software (H7, H14).
- **The relay adds about 7 microseconds of work per packet** (0.17% of the
  4 ms between packets, measured on an M3 Max). The delay that matters is the
  Bluetooth link's own, the same on any host; `delivery_delay_ms` in the
  report shows how late packets arrive relative to the best ones.
- **No reconnect.** If the link is lost or goes silent for 3 s the relay
  exits 4 and the panel shows "EEG stream lost". Restart it.
- The port is opened exclusively: a second program reading it would steal
  bytes and look like packet loss.

Try it with no headset: `python tools/fake_unicorn_bt.py --port 9000`, then
`python collection/unicorn_relay.py --tcp 127.0.0.1:9000`. The relay
recognises the fake's marker and names its streams `MockUnicornEEG` /
`MockUnicornAux`, so synthetic data cannot pass for a real `UnicornEEG`.

Exit codes: 0 stopped, 2 bad arguments or no headset port, 3 could not
connect, 4 link lost or silent.

## Startup sequence -- order matters

1. **Start the Unicorn LSL Interface** (Windows; on a Mac start
   `collection/unicorn_relay.py` instead, see above). Connect to the headset, click
   Start streaming. Confirm that it is actively streaming
   before moving on.
2. **Start `collection/record_session.py`.** It resolves the
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

Start the Unicorn LSL Interface and click Start. Then, on the same machine,
the normal path is the session panel. It starts the recorder, waits until
the recorder's log reports `recording_started`, and only then starts the
stimulus:

```bash
./start_speller.sh
```

`start_speller.sh` is a bash script. On the Windows deployment machine, run
it from Git Bash or WSL, or call the two processes below directly from
whatever shell you use there.

To run the two by hand, call the environment's interpreter by path rather
than a bare `python` -- on the Mac this repo is developed on, a pipenv
virtualenv earlier on `PATH` shadows `python` even after `conda activate`;
check whether the deployment machine's own environment has an equivalent
shadowing issue before assuming it doesn't.

```bash
PY="$(conda info --base)/envs/p300speller/bin/python"   # adjust for the deployment machine's env
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
timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, marker_json_or_empty,
trial_id, flash_idx, group_json, is_target, target_char_idx
```

- `timestamp`: LSL timestamp, mapped into this machine's `local_clock()`
  domain (the inlets use `proc_clocksync`). On one machine the correction
  is ~0, but the mapping stays on unconditionally: without it, an EEG
  source on a different clock (the old split-machine variant, or any
  future setup) would leave timestamps unaligned with local markers.
  Whether EEG also uses `proc_dejitter` is
  one switch shared with the inference service:
  `pipeline/constants.EEG_INLET_DEJITTER` (currently off, so integrity
  checks see timestamps as delivered). The effective flags are recorded
  in the log and the integrity summary.
- `Fz`...`PO8`: the 8 Unicorn channels (`pipeline.constants.EEG_CHANNELS`), one row per EEG sample (250 Hz).
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
  code 3 after `--eeg-timeout`** -- the Unicorn LSL Interface either isn't
  running, or isn't actually streaming (check it says "Streaming" in its
  own UI, not just "Connected"). LSL discovery uses UDP multicast by
  default; on one machine this is loopback traffic and essentially never
  the cause. (If you're running the split-machine variant instead, also
  check the VM's network mode and firewall -- multicast across different
  virtual network segments can silently fail even when the other machine
  is otherwise reachable.)
- **Sanity check reports all channels flat (~0 uV)** -- electrodes not
  making contact, or the Unicorn isn't actually powered on despite the
  LSL Interface showing "Streaming."
- **Sanity check reports amplitudes in the thousands/millions** -- almost
  certainly a units bug somewhere in the chain (see docs/CLAUDE.md gotcha
  #1 for the exact failure mode this project has hit before with a
  similar units mismatch), not real signal.
