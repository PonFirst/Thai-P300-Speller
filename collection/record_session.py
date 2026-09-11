"""
LSL-consuming EEG recorder for the Thai P300 speller.

Does NOT talk to the Unicorn Hybrid Black directly -- no BrainFlow, no
Unicorn SDK calls. BrainFlow's UnicornBoard class is compiled out on
macOS entirely (unicorn_board.h is wrapped in
`#if defined __linux__ || defined _WIN32`), and g.tec's own Unicorn
Suite / Unicorn LSL Interface is Windows-only, so on this project's
Apple Silicon development machine the only viable path is: run g.tec's
official Unicorn LSL Interface on a Windows-on-ARM VM (Parallels, with
the Unicorn's USB dongle passed through) or on Linux, which publishes an
LSL outlet, and have this script pull from that outlet over the network/
loopback. This script is a pure LSL consumer -- see README_ACQUISITION.md
for the full machine topology and startup order.

Consumes two LSL streams per the shared interface contract:
  EEG stream:    name='UnicornEEG', type='EEG', channel_count=8,
                 sample_rate=250, dtype=float32
  Marker stream: name='P300Markers', type='Markers', channel_count=1,
                 sample_rate=0 (irregular), dtype=string, one
                 JSON-encoded event per sample (see
                 collection/checkerboard_stimulus.py for the producer).

Writes one CSV per session with columns:
  timestamp, Fz, F3, C4, Cz, Pz, P3, O2, O1, marker_json_or_empty,
  trial_id, flash_idx, group_json, is_target, target_char_idx

CSV (not Parquet): chosen so the file can be opened/inspected by hand or
in Excel mid-collection without a Python session, and so a crash mid-write
loses at most the last buffered rows rather than corrupting a whole
row-group -- append-friendly plain text beats Parquet's write-once-per-
row-group model for a script that streams rows continuously for an
unknown session length. Switch to Parquet later for analysis-time
compaction if file size becomes a problem; it is not a problem at
8ch x 250Hz for a single session.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
from pylsl import StreamInlet, resolve_byprop

EEG_CHANNELS = ["Fz", "F3", "C4", "Cz", "Pz", "P3", "O2", "O1"]
CSV_COLUMNS = (
    ["timestamp"] + EEG_CHANNELS +
    ["marker_json_or_empty", "trial_id", "flash_idx", "group_json",
     "is_target", "target_char_idx"]
)

# Plausible EEG amplitude range, peak-to-peak, in microvolts. Real scalp
# EEG is roughly single digits to a few hundred uV; well outside this
# range means a railed/clipped/flat/disconnected channel, not real signal.
SANITY_PTP_MIN_UV = 1.0
SANITY_PTP_MAX_UV = 200.0
SANITY_CHECK_DURATION_S = 2.0


def resolve_eeg_inlet(timeout: float = 5.0) -> StreamInlet:
    """
    Find the EEG outlet published by the Unicorn LSL Interface (running
    on the Windows VM / Linux box -- see README_ACQUISITION.md). Retries
    are the caller's responsibility (see wait_for_streams below); a
    single resolve_byprop call here just does one timeout-bounded probe.
    """
    streams = resolve_byprop("type", "EEG", timeout=timeout)
    if not streams:
        return None
    return StreamInlet(streams[0])


def resolve_marker_inlet(timeout: float = 5.0) -> StreamInlet:
    """
    Find the marker outlet published by collection/checkerboard_stimulus.py.
    That script may not have started yet even after this recorder has --
    see wait_for_streams() for the polling/retry loop that tolerates that.
    """
    streams = resolve_byprop("type", "Markers", timeout=timeout)
    if not streams:
        return None
    return StreamInlet(streams[0])


def wait_for_streams(poll_interval_s: float = 3.0, max_wait_s: float = 300.0):
    """
    Poll for both the EEG and marker outlets, retrying rather than
    failing immediately if either isn't up yet. The marker outlet in
    particular is allowed to appear AFTER this script starts (the
    stimulus script is started last in the acquisition sequence -- see
    README_ACQUISITION.md) -- so this keeps trying instead of giving up
    on the first empty resolve_byprop() call.

    Returns:
        (eeg_inlet, marker_inlet)

    Raises:
        TimeoutError if either stream hasn't appeared within max_wait_s.
    """
    eeg_inlet = None
    marker_inlet = None
    waited = 0.0

    while waited <= max_wait_s:
        if eeg_inlet is None:
            eeg_inlet = resolve_eeg_inlet(timeout=1.0)
            if eeg_inlet is not None:
                print("[record_session] Found EEG outlet ('UnicornEEG').")
        if marker_inlet is None:
            marker_inlet = resolve_marker_inlet(timeout=1.0)
            if marker_inlet is not None:
                print("[record_session] Found marker outlet ('P300Markers').")

        if eeg_inlet is not None and marker_inlet is not None:
            return eeg_inlet, marker_inlet

        missing = []
        if eeg_inlet is None:
            missing.append("EEG ('UnicornEEG')")
        if marker_inlet is None:
            missing.append("Markers ('P300Markers')")
        print(f"[record_session] Waiting for stream(s): {', '.join(missing)} "
              f"-- retrying in {poll_interval_s:.0f}s "
              f"({waited:.0f}s / {max_wait_s:.0f}s elapsed)...")
        time.sleep(poll_interval_s)
        waited += poll_interval_s

    missing = []
    if eeg_inlet is None:
        missing.append("EEG")
    if marker_inlet is None:
        missing.append("Markers")
    raise TimeoutError(
        f"Timed out after {max_wait_s:.0f}s waiting for stream(s): "
        f"{', '.join(missing)}. Confirm the Unicorn LSL Interface "
        f"(and, for Markers, checkerboard_stimulus.py) are actually "
        f"running -- see README_ACQUISITION.md for the required startup order."
    )


def run_signal_sanity_check(eeg_inlet: StreamInlet,
                             duration_s: float = SANITY_CHECK_DURATION_S) -> bool:
    """
    Pull `duration_s` seconds of real EEG data before the session starts
    and check per-channel peak-to-peak amplitude falls in a plausible
    microvolt range. Warns LOUDLY (does not silently continue) on
    clipped/flat/railed data -- a silent pass-through here is exactly how
    a disconnected electrode or wrong-units stream goes unnoticed until
    the whole session is unusable.

    Returns:
        True if all channels look plausible, False otherwise (caller
        decides whether to abort -- this function only checks and warns).
    """
    print(f"[record_session] Running {duration_s:.0f}s signal sanity check...")
    samples = []
    t_start = time.time()
    while time.time() - t_start < duration_s:
        chunk, _ = eeg_inlet.pull_chunk(timeout=0.5)
        if chunk:
            samples.extend(chunk)

    if not samples:
        print("[record_session] *** WARNING: no EEG samples received during "
              "sanity check. Is the Unicorn LSL Interface actually streaming? ***")
        return False

    arr = np.asarray(samples, dtype=float)  # shape (n_samples, n_channels)
    n_channels = min(arr.shape[1], len(EEG_CHANNELS))
    ok = True
    for ch_idx in range(n_channels):
        ch_name = EEG_CHANNELS[ch_idx]
        col = arr[:, ch_idx]
        ptp_uv = float(col.max() - col.min())
        if ptp_uv < SANITY_PTP_MIN_UV:
            print(f"[record_session] *** WARNING: channel {ch_name} peak-to-peak "
                  f"= {ptp_uv:.4f} uV -- looks FLAT/disconnected "
                  f"(expected {SANITY_PTP_MIN_UV}-{SANITY_PTP_MAX_UV} uV). ***")
            ok = False
        elif ptp_uv > SANITY_PTP_MAX_UV:
            print(f"[record_session] *** WARNING: channel {ch_name} peak-to-peak "
                  f"= {ptp_uv:.4f} uV -- looks CLIPPED/RAILED or in the wrong "
                  f"units (expected {SANITY_PTP_MIN_UV}-{SANITY_PTP_MAX_UV} uV). ***")
            ok = False
        else:
            print(f"[record_session]   {ch_name}: {ptp_uv:.2f} uV p-p (OK)")

    if ok:
        print("[record_session] Signal sanity check passed.")
    else:
        print("[record_session] *** Signal sanity check FAILED for one or more "
              "channels -- fix electrode contact / stream units before "
              "collecting real data. Continuing anyway (not aborted "
              "automatically), but do not trust this session's data "
              "without investigating the warning(s) above. ***")
    return ok


def parse_marker_json(marker_str: str) -> dict:
    """
    Parse one marker sample's JSON string into the shared-contract event
    dict. Returns {} on empty/unparseable input rather than raising --
    a single malformed marker sample should not crash a multi-hour
    recording session; it just becomes a row with all marker fields empty.
    """
    if not marker_str:
        return {}
    try:
        return json.loads(marker_str)
    except (json.JSONDecodeError, TypeError):
        print(f"[record_session] *** WARNING: unparseable marker JSON, "
              f"skipping: {marker_str!r} ***")
        return {}


def event_to_row_fields(event: dict) -> dict:
    """
    Map one parsed marker event dict to this CSV's parsed marker columns.
    Fields not present in a given event type (e.g. group_json for a
    trial_start/trial_end event) are left empty.
    """
    return {
        "trial_id": event.get("trial_id", ""),
        "flash_idx": event.get("flash_idx", ""),
        "group_json": json.dumps(event["group"]) if "group" in event else "",
        "is_target": event.get("is_target", ""),
        "target_char_idx": event.get("target_char_idx", ""),
    }


def merge_by_nearest_timestamp(eeg_samples: list, eeg_timestamps: list,
                                marker_samples: list, marker_timestamps: list) -> list:
    """
    Merge EEG samples and marker events into output rows by nearest
    timestamp: every EEG sample becomes one row, and any marker whose
    timestamp is closer to that EEG sample than to its neighbors is
    attached to it. Markers arrive far less often than EEG samples (one
    per flash vs. 250/sec), so this is effectively "attach each marker
    to the EEG sample closest in time to it," with all other rows
    carrying an empty marker.

    Returns:
        list of dicts, each with keys timestamp, <EEG_CHANNELS...>,
        marker_json_or_empty, trial_id, flash_idx, group_json, is_target,
        target_char_idx -- one dict per EEG sample, in timestamp order.
    """
    rows = []
    for ts, sample in zip(eeg_timestamps, eeg_samples):
        row = {"timestamp": ts}
        for ch_idx, ch_name in enumerate(EEG_CHANNELS):
            row[ch_name] = sample[ch_idx] if ch_idx < len(sample) else ""
        row["marker_json_or_empty"] = ""
        row.update({"trial_id": "", "flash_idx": "", "group_json": "",
                    "is_target": "", "target_char_idx": ""})
        rows.append(row)

    if not rows or not marker_timestamps:
        return rows

    eeg_ts_arr = np.asarray(eeg_timestamps)
    for m_ts, m_sample in zip(marker_timestamps, marker_samples):
        nearest_idx = int(np.argmin(np.abs(eeg_ts_arr - m_ts)))
        event = parse_marker_json(m_sample[0] if m_sample else "")
        rows[nearest_idx]["marker_json_or_empty"] = m_sample[0] if m_sample else ""
        rows[nearest_idx].update(event_to_row_fields(event))

    return rows


def record(out_path: str, session_duration_s: float | None = None,
           poll_interval_s: float = 3.0, max_wait_s: float = 300.0,
           chunk_pull_timeout_s: float = 0.2,
           skip_sanity_check: bool = False):
    """
    Main recording loop: resolve both LSL outlets, run the pre-session
    signal sanity check, then continuously pull chunks from both streams,
    merge them, and append rows to `out_path` until interrupted
    (Ctrl+C) or session_duration_s elapses (None = run until interrupted).
    """
    eeg_inlet, marker_inlet = wait_for_streams(
        poll_interval_s=poll_interval_s, max_wait_s=max_wait_s)

    if not skip_sanity_check:
        run_signal_sanity_check(eeg_inlet)
    else:
        print("[record_session] Skipping signal sanity check (--skip-sanity-check).")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[record_session] Recording to {out_path} -- press Ctrl+C to stop.")

    n_rows_written = 0
    t_start = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        try:
            while session_duration_s is None or (time.time() - t_start) < session_duration_s:
                eeg_chunk, eeg_ts = eeg_inlet.pull_chunk(timeout=chunk_pull_timeout_s)
                marker_chunk, marker_ts = marker_inlet.pull_chunk(timeout=0.0)

                if not eeg_chunk:
                    continue

                rows = merge_by_nearest_timestamp(eeg_chunk, eeg_ts, marker_chunk, marker_ts)
                for row in rows:
                    writer.writerow(row)
                n_rows_written += len(rows)
                f.flush()
        except KeyboardInterrupt:
            print("\n[record_session] Ctrl+C received -- stopping.")

    print(f"[record_session] Done. Wrote {n_rows_written} rows to {out_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Record a Thai P300 speller session from existing "
                    "LSL EEG + marker outlets (does not talk to hardware directly).")
    parser.add_argument("--out", type=str, default=None,
                         help="Output CSV path (default: "
                              "collection/recordings/session_<timestamp>.csv).")
    parser.add_argument("--duration", type=float, default=None,
                         help="Session duration in seconds (default: run until Ctrl+C).")
    parser.add_argument("--poll-interval", type=float, default=3.0,
                         help="Seconds between stream-resolution retries.")
    parser.add_argument("--max-wait", type=float, default=300.0,
                         help="Max seconds to wait for both LSL outlets before giving up.")
    parser.add_argument("--skip-sanity-check", action="store_true",
                         help="Skip the pre-session signal amplitude sanity check "
                              "(not recommended -- for debugging only).")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            os.path.dirname(__file__), "recordings", f"session_{timestamp}.csv")

    record(out_path, session_duration_s=args.duration,
           poll_interval_s=args.poll_interval, max_wait_s=args.max_wait,
           skip_sanity_check=args.skip_sanity_check)


if __name__ == "__main__":
    main()
