#!/bin/bash
# Show the Thai checkerboard speller in BCI2000 with no headset.
#
#   bci2000/demo_speller.sh                  # spells สวัสดีครับ at the real flash speed
#   bci2000/demo_speller.sh --slow           # 0.5 s flashes, to look at each lit group
#   bci2000/demo_speller.sh --slow ภาษาไทย   # your own text (every character must be in the matrix)
#
# Starts tools/fake_unicorn_bt.py (SYNTHETIC EEG) and BCI2000 with UnicornSerialSource reading it,
# P3SignalProcessing and CheckerboardSpeller, loads parms/ThaiCheckerboard.prm in copy mode and
# runs until the text is done. Ctrl+C stops everything. The recording goes to a temp folder
# (printed at the end), never into the repo.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(conda info --base)/envs/p300speller/bin/python"
PROG="${BCI2000_ROOT:-$HOME/BCI2000/trunk}/prog"
PORT=9000

SLOW=0
if [ "${1:-}" = "--slow" ]; then SLOW=1; shift; fi
TEXT="${1:-สวัสดีครับ}"

for m in BCI2000Shell UnicornSerialSource.app P3SignalProcessing.app CheckerboardSpeller.app; do
  [ -e "$PROG/$m" ] || { echo "Missing $PROG/$m: build first (bci2000/README.md)"; exit 1; }
done
if pgrep -f "$PROG/.*(Source|ADC|Operator)" > /dev/null; then
  echo "BCI2000 is already running. Close it first (or: pkill -f '$PROG/')."; exit 1
fi
if lsof -i ":$PORT" > /dev/null 2>&1; then
  echo "Port $PORT is busy (an old fake headset?). Stop it first: pkill -f fake_unicorn_bt"; exit 1
fi

ENCODED="$("$PY" -c "import sys; sys.path.insert(0, '$REPO/tools'); from make_bci2000_parms import encode, load_characters
t = sys.argv[1]; bad = [c for c in t if c not in load_characters()]
sys.exit('Not in the matrix: ' + ' '.join(bad)) if bad else print(encode(t))" "$TEXT")"

WORK="$(mktemp -d -t thai_speller_demo)"
"$PY" "$REPO/tools/fake_unicorn_bt.py" --port "$PORT" > "$WORK/fake_headset.log" 2>&1 &
FAKE_PID=$!
cleanup() {
  # BCI2000 first: a source module that loses the fake headset mid-shutdown reports a lost link.
  pkill -f "$PROG/" 2> /dev/null || true
  sleep 1
  kill "$FAKE_PID" 2> /dev/null || true
}
trap cleanup EXIT INT TERM
sleep 1

TIMING=""
if [ "$SLOW" = 1 ]; then
  TIMING="Set parameter StimulusDuration 500ms
Set parameter ISIMinDuration 500ms
Set parameter ISIMaxDuration 500ms"
fi

cat > "$WORK/demo.bat" << EOF
Change directory \$BCI2000LAUNCHDIR
Reset system
Startup system localhost --SystemLogFile=$WORK/system.log
Start executable UnicornSerialSource --local --TcpAddress=127.0.0.1:$PORT
Start executable P3SignalProcessing --local
Start executable CheckerboardSpeller --local
Wait for Connected 30
Load parameterfile "$REPO/bci2000/parms/ThaiCheckerboard.prm"
Set parameter DataDirectory $WORK/data
Set parameter SubjectName demo
Set parameter InterpretMode 2
Set parameter TextToSpell $ENCODED
Set parameter NumberOfSequences 2
Set parameter StoppingRule 0
Set parameter WindowLeft 0
Set parameter WindowTop 40
Set parameter WindowWidth 1280
Set parameter WindowHeight 800
$TIMING
Set config
Wait for Resting 30
Sleep 2
Start
Wait for Suspended 900
Quit
EOF

echo "SYNTHETIC EEG from the fake headset; the classifier is untrained, so the spelled text is random."
echo "Spelling: $TEXT   (each character is shown in green first, then the groups flash)"
cd "$PROG"
env -u QT_QPA_PLATFORM ./BCI2000Shell "$WORK/demo.bat" || echo "BCI2000 stopped early: see the errors below."
echo
echo "Done. Recording and speller log: $WORK/data   BCI2000 system log: $WORK/system.log"
if grep -q -i -E "error|warning" "$WORK/system.log" 2> /dev/null; then
  echo "--- BCI2000 errors and warnings ---"
  grep -i -E "error|warning" "$WORK/system.log"
fi
cat "$WORK"/data/*/*.applog 2> /dev/null | grep -E "Selection|Run ended" || true
