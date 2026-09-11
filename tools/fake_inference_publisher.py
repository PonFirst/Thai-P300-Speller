"""
Scripted P300Inference publisher for manual QA of the GUI -- no headset,
classifier, or stopping controller required.

Every event goes through pipeline.events.encode(), so the GUI receives exactly
what the real inference service would send. The script covers all three commit
reasons (threshold_crossed, max_flashes_reached, trial_ended), a quality-gate
rejection, and optionally one malformed payload. When ก, the mai ek tone mark,
and อ are all in the matrix, the committed characters compose to "ก่อ", which
also exercises combining-mark rendering.

    python tools/fake_inference_publisher.py                        # fingerprint from session_manifest.json, if valid
    python tools/fake_inference_publisher.py --manifest path/to/session_manifest.json
    python tools/fake_inference_publisher.py --fingerprint 0000     # force a manifest mismatch in the GUI
    python tools/fake_inference_publisher.py --include-malformed --flash-interval 0.02
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pylsl import StreamInfo, StreamOutlet  # noqa: E402

from pipeline import events as ev  # noqa: E402
from pipeline.constants import (  # noqa: E402
    INFERENCE_FINGERPRINT_KEY,
    INFERENCE_STREAM_NAME,
    INFERENCE_STREAM_TYPE,
)
from pipeline.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_STOPPING,
    ManifestError,
    load_characters_file,
    load_manifest,
)

CHARACTERS_PATH = REPO_ROOT / "collection" / "characters_th.json"


def pick_targets(characters: list[str]) -> list[int]:
    wanted = ["ก", "่", "อ"]
    if all(c in characters for c in wanted):
        return [characters.index(c) for c in wanted]
    print("note: ก / ่ / อ are not all in the matrix; using its first three characters")
    return [0, 1, 2]


def _ramp(start: float, end: float, n: int) -> list[float]:
    return [start + (end - start) * i / max(n - 1, 1) for i in range(n)]


def _trial(trial_id, target, n_updates, end_prob, reason, theta, characters, interval,
           reject_at=None):
    """Posterior updates climbing to end_prob, then one commit. Returns (payload, pause) pairs."""
    steps = []
    decoy = (target + 5) % len(characters)
    probs = _ramp(0.05, end_prob, n_updates)
    flash_idx = 0
    for i, prob in enumerate(probs):
        if reject_at is not None and flash_idx == reject_at:
            steps.append((ev.encode(ev.flash_rejected(trial_id, flash_idx, ev.REJECT_QUALITY_GATE, ["Pz"])),
                          interval))
            flash_idx += 1
        leading = decoy if i < n_updates // 3 else target
        steps.append((ev.encode(ev.posterior_update(trial_id, flash_idx, prob, leading, theta)), interval))
        flash_idx += 1
    committed = ev.character_committed(trial_id, target, characters[target], n_updates, reason,
                                       characters=characters)
    steps.append((ev.encode(committed, characters=characters), 1.5))
    return steps


def build_script(characters, fingerprint, theta, t_max, interval, include_malformed):
    first, second, third = pick_targets(characters)
    script = [
        (ev.encode(ev.service_status(ev.STATE_STARTING, "fake publisher starting", fingerprint)), 0.5),
        (ev.encode(ev.service_status(ev.STATE_READY, "fake publisher ready", fingerprint)), 0.5),
    ]
    script += _trial(0, first, 10, min(0.97, theta + 0.03), ev.REASON_THRESHOLD_CROSSED,
                     theta, characters, interval, reject_at=3)
    if include_malformed:
        script.append(('{"event": "posterior_update", "trial_id": 1}', 0.5))
    script += _trial(1, second, t_max, 0.6, ev.REASON_MAX_FLASHES_REACHED, theta, characters, interval)
    script += _trial(2, third, 60, 0.45, ev.REASON_TRIAL_ENDED, theta, characters, interval)
    script.append((ev.encode(ev.service_status(ev.STATE_STOPPED, "fake publisher finished", fingerprint)), 0.0))
    return script


def resolve_settings(args):
    theta, t_max = DEFAULT_STOPPING["theta"], DEFAULT_STOPPING["t_max"]
    fingerprint = args.fingerprint
    manifest_path = Path(args.manifest) if args.manifest else DEFAULT_MANIFEST_PATH
    try:
        manifest = load_manifest(manifest_path)
        theta, t_max = manifest.stopping["theta"], manifest.stopping["t_max"]
        if fingerprint is None:
            fingerprint = manifest.fingerprint
    except ManifestError as exc:
        print(f"warning: {exc}")
        if fingerprint is None:
            print("         publishing without a manifest fingerprint; the GUI should flag this")
    return fingerprint, theta, t_max


def main():
    parser = argparse.ArgumentParser(description="Publish a scripted P300Inference stream for GUI QA.")
    parser.add_argument("--manifest", default=None,
                        help=f"manifest to take the fingerprint and theta/t_max from (default {DEFAULT_MANIFEST_PATH})")
    parser.add_argument("--fingerprint", default=None,
                        help="override the published fingerprint (use a wrong one to test mismatch handling)")
    parser.add_argument("--flash-interval", type=float, default=0.124,
                        help="seconds between events within a trial (default 0.124, the real SOA)")
    parser.add_argument("--include-malformed", action="store_true",
                        help="push one schema-violating payload after the first commit")
    parser.add_argument("--wait", type=float, default=30.0,
                        help="seconds to wait for a consumer (the GUI) before publishing")
    parser.add_argument("--inference-type", default=INFERENCE_STREAM_TYPE, help="LSL stream type")
    args = parser.parse_args()

    characters = load_characters_file(CHARACTERS_PATH)
    fingerprint, theta, t_max = resolve_settings(args)

    info = StreamInfo(INFERENCE_STREAM_NAME, args.inference_type, 1, 0, "string", "fake_inference_publisher")
    info.desc().append_child_value(INFERENCE_FINGERPRINT_KEY, fingerprint or "")
    outlet = StreamOutlet(info)

    print(f"publishing '{INFERENCE_STREAM_NAME}' (type {args.inference_type}), "
          f"fingerprint {fingerprint[:12] if fingerprint else '<none>'}, theta {theta}, t_max {t_max}")
    if not outlet.wait_for_consumers(args.wait):
        print(f"no consumer connected within {args.wait:.0f}s; publishing anyway")

    for payload, pause in build_script(characters, fingerprint, theta, t_max,
                                       args.flash_interval, args.include_malformed):
        outlet.push_sample([payload])
        time.sleep(pause)
    print("done")


if __name__ == "__main__":
    main()
