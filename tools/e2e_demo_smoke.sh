#!/usr/bin/env bash
#
# Hardware-free end-to-end smoke test:
#   mock headset -> real recorder + real PsychoPy stimulus -> fit on the recorded CSV
#   -> inference service reaches "ready" on that model -> P300 check at Pz.
#
# Opens a small stimulus window for about a minute. Writes only to a fresh temp
# directory. The stimulus publishes on the production marker type ("Markers"),
# so don't run this while a real session is running.
#
# Usage: tools/e2e_demo_smoke.sh        (exit 0 = every stage passed)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$(conda info --base)/envs/p300speller/bin/python}"
OUT="$(mktemp -d -t p300_e2e)"
EEG_TYPE="EEGE2E_$(uuidgen | tr -d -)"
CSV="$OUT/demo_e2e.csv"
MOCK= REC= SVC=

cleanup() {
    for pid in $SVC $REC $MOCK; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

echo "output directory: $OUT"
cd "$ROOT/collection"
"$PY" mock_headset.py --eeg-type "$EEG_TYPE" --marker-type Markers > "$OUT/mock.out" 2>&1 & MOCK=$!
sleep 3
"$PY" record_session.py --out "$CSV" --eeg-type "$EEG_TYPE" --marker-type Markers > "$OUT/recorder.out" 2>&1 & REC=$!
for _ in $(seq 1 120); do
    grep -q '"event": "recording_started"' "$CSV.log.jsonl" 2>/dev/null && break
    sleep 0.5
done
"$PY" checkerboard_stimulus.py --targets "กขค" --n-repetitions 3 --cue-ms 1000 --windowed > "$OUT/stimulus.out" 2>&1
STIM_RC=$?
sleep 3
kill -TERM "$REC"; wait "$REC"; REC_RC=$?; REC=

cd "$ROOT"
"$PY" -m pipeline.fit_from_recording --recording "$CSV" --id e2e --allow-demo \
    --models-dir "$OUT/models" --manifest "$OUT/demo_session_manifest.json" > "$OUT/fit.out" 2>&1
FIT_RC=$?

SVC_RC=1
if [ "$FIT_RC" -eq 0 ]; then
    "$PY" -m pipeline.inference_service --manifest "$OUT/demo_session_manifest.json" \
        --eeg-type "$EEG_TYPE" --marker-type "MKE2E_$(uuidgen | tr -d -)" \
        --inference-type "INFE2E_$(uuidgen | tr -d -)" --log-file "$OUT/service.log.jsonl" \
        > /dev/null 2> "$OUT/service.err" & SVC=$!
    READY=0
    for _ in $(seq 1 120); do
        grep -q '"state": "ready"' "$OUT/service.log.jsonl" 2>/dev/null && READY=1 && break
        sleep 0.5
    done
    kill -TERM "$SVC"; wait "$SVC"; STOP_RC=$?; SVC=
    [ "$READY" -eq 1 ] && [ "$STOP_RC" -eq 0 ] && SVC_RC=0
fi

"$PY" - "$CSV" <<'PYEOF'
import json, sys
import numpy as np
from pipeline.epoching import EpochSpec, extract_epochs
from pipeline.fit_from_recording import load_recording

csv = sys.argv[1]
rec = load_recording(csv)
spec = EpochSpec(tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0))
X, kept, _ = extract_epochs(rec.timestamps, rec.data, [f.onset for f in rec.flashes], spec)
y = np.array([f.is_target for f, k in zip(rec.flashes, kept) if k])
pz = spec.channels.index("Pz")
t = spec.tmin + np.arange(spec.n_times) / spec.sfreq
w = (t >= 0.25) & (t <= 0.55)
target = X[y, pz].mean(0)
summary = json.load(open(csv + ".integrity.json"))
steps = np.diff(rec.timestamps)
print(f"recording: {len(rec.timestamps)} EEG samples, {len(rec.flashes)} flashes, "
      f"{int((steps <= 0).sum())} non-monotonic steps, issues {summary['issues']}")
print(f"Pz target average peak {target[w].max():+.2f} uV at {t[w][target[w].argmax()] * 1000:.0f} ms "
      "(the mock injects about +6 uV at ~350 ms)")
PYEOF

echo "--- fit ---"; cat "$OUT/fit.out"
echo "EXIT CODES: stimulus=$STIM_RC recorder=$REC_RC fit=$FIT_RC service=$SVC_RC"
[ "$STIM_RC" -eq 0 ] && [ "$REC_RC" -eq 0 ] && [ "$FIT_RC" -eq 0 ] && [ "$SVC_RC" -eq 0 ]
