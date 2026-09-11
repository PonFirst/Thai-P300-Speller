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
2. **Start `collection/record_session.py`** on this Mac.
   `record_session.py` calls `pylsl.resolve_byprop('type', 'EEG',
   timeout=5)` to find the `'UnicornEEG'` outlet from step 1 -- this
   must already exist, or the resolve call (and the retry loop wrapped
   around it) will keep polling and printing "waiting for stream(s)"
   until it does. It ALSO calls `resolve_byprop('type', 'Markers',
   timeout=5)` for the `'P300Markers'` outlet from step 3, but tolerates
   that one not existing yet -- it polls/retries rather than failing
   immediately, since step 3 is started after this script.
3. **Start `collection/checkerboard_stimulus.py`** (the PsychoPy
   stimulus window) on this Mac. Once it opens its LSL outlet,
   `record_session.py`'s marker-stream poll will find it and recording
   proper begins.

**Why this exact order**: `record_session.py` needs BOTH outlets to
exist before it can resolve them and start merging rows. Starting the
EEG source first is required -- there is no retry path around "the
headset isn't streaming at all." Starting the stimulus script last is a
convenience, not a hard requirement: `record_session.py`'s marker-outlet
resolution is a poll/retry loop specifically so you *can* start it before
the stimulus script without a crash, but starting the recorder before
the EEG source is not similarly tolerated (that step 1 -> step 2
ordering matters most).

## Running it

On the Windows VM / Linux box: start the Unicorn LSL Interface and click
Start.

On this Mac:

```bash
conda activate p300speller
python collection/record_session.py --out collection/recordings/session_001.csv
```

Then, in a second terminal on this Mac, once `record_session.py` prints
that it's waiting for the marker stream:

```bash
conda activate p300speller
python collection/checkerboard_stimulus.py
```

Stop recording with Ctrl+C in the `record_session.py` terminal (or pass
`--duration <seconds>` to stop automatically).

## Output format

One CSV per session (see `record_session.py`'s module docstring for why
CSV over Parquet), columns:

```
timestamp, Fz, F3, C4, Cz, Pz, P3, O2, O1, marker_json_or_empty,
trial_id, flash_idx, group_json, is_target, target_char_idx
```

- `timestamp`: LSL local_clock() timestamp, common to both streams
  (the whole point of using LSL is a shared clock domain).
- `Fz`...`O1`: the 8 Unicorn channels, one row per EEG sample (250 Hz).
- `marker_json_or_empty`: the raw JSON string pushed by
  `checkerboard_stimulus.py` for this row if a marker landed closest to
  this EEG sample's timestamp, else empty.
- `trial_id`, `flash_idx`, `group_json`, `is_target`, `target_char_idx`:
  the same marker JSON pre-parsed into columns, so downstream analysis
  never has to re-parse the JSON string. Empty for every non-marker row.
  `group_json` is the full list of character indices in the flashed
  group -- this is the field the original reference implementation never
  logged, and its absence is what made that data useless for replay
  (see docs/CLAUDE.md and stopping/replay.py). Do not regress this.

## Pre-session signal sanity check

`record_session.py` pulls 2 seconds of real EEG data before the actual
session starts and checks each channel's peak-to-peak amplitude falls
in a plausible range (roughly 1-200 uV). It warns LOUDLY (printed
`*** WARNING ***` lines, not a silent pass-through) if a channel looks
flat/disconnected or clipped/railed/wrong-units. This does not abort the
session automatically -- investigate any warning before trusting that
session's data, don't just proceed past it.

## Troubleshooting

- **`record_session.py` hangs on "Waiting for stream(s): EEG
  ('UnicornEEG')"** -- the Unicorn LSL Interface on the VM/Linux box
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
