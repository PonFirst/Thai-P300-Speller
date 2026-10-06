#!/bin/bash
# Run one BCI2000 session run of the Thai checkerboard speller, configured the same way every time.
#
#   bci2000/run_session.sh calibrate --subject S01 --text "ภาษาไทย" [--reps 5]
#   bci2000/run_session.sh online    --subject S01 --model models/S01_bci2000.prm --text "สวัสดี"
#                                    [--condition transformer|uniform|fixed] [--reps 8]
#   bci2000/run_session.sh online    --subject S01 --model models/S01_bci2000.prm --free
#   bci2000/run_session.sh check     --subject S01        # 60 s: blink / eyes closed / jaw, for H3 + H13
#
# Common options: --session 001 (BCI2000 SubjectSession), --data DIR (default collection/recordings/bci2000),
#   --window 1280x800, --fake (rehearse with tools/fake_unicorn_bt.py; the data marks itself synthetic),
#   --auto (start the run by itself 2 s after configuring, instead of waiting for Start: rehearsals, tests).
#
# Modules: UnicornSerialSource always; calibrate and check use DummySignalProcessing (no classifier),
# online uses CheckerboardSignalProcessing with the fitted model. Conditions (online):
#   transformer  Bayesian stopping + the Transformer prior (the model .prm's own settings; the default)
#   uniform      Bayesian stopping + uniform prior (the no-LM control)
#   fixed        no stopping: every flash of NumberOfSequences repetitions, the best mean score
#
# The Operator window opens configured: check the signal in the Source window, then press Start.
# The script ends when that run ends; run it again for the next run (BCI2000 numbers runs R01, R02, ...).
# The parameters every run used are stored in its .dat, so the condition is always recoverable.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(conda info --base)/envs/p300speller/bin/python"
PROG="${BCI2000_ROOT:-$HOME/BCI2000/trunk}/prog"
SPELLER_PRM="$REPO/bci2000/parms/ThaiCheckerboard.prm"
PORT=9000

die() { echo "run_session: $*" >&2; exit 2; }
usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

MODE="${1:-}"; shift || true
case "$MODE" in calibrate|online|check) ;; *) usage ;; esac

SUBJECT="" SESSION="001" TEXT="" MODEL="" CONDITION="transformer" REPS="" FREE=0 FAKE=0 AUTO=0
DATA="$REPO/collection/recordings/bci2000" WINDOW="1280x800"
while [ $# -gt 0 ]; do
  case "$1" in
    --subject) SUBJECT="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --text) TEXT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --condition) CONDITION="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --free) FREE=1; shift ;;
    --fake) FAKE=1; shift ;;
    --auto) AUTO=1; shift ;;
    --data) DATA="$2"; shift 2 ;;
    --window) WINDOW="$2"; shift 2 ;;
    *) die "unknown option $1 (see: $0 with no arguments)" ;;
  esac
done

[[ "$SUBJECT" =~ ^[A-Za-z0-9]+$ ]] || die "--subject must be letters and digits only (BCI2000 builds file names from it)"
[[ "$SESSION" =~ ^[0-9]{1,3}$ ]] || die "--session must be a number of up to 3 digits"
[[ "$WINDOW" =~ ^[0-9]+x[0-9]+$ ]] || die "--window must look like 1280x800"
[ -z "$REPS" ] || [[ "$REPS" =~ ^[1-9][0-9]*$ ]] || die "--reps must be a positive integer"
case "$CONDITION" in transformer|uniform|fixed) ;; *) die "--condition must be transformer, uniform or fixed" ;; esac

for m in BCI2000Shell UnicornSerialSource.app DummySignalProcessing.app CheckerboardSignalProcessing.app \
         CheckerboardSpeller.app DummyApplication.app; do
  [ -e "$PROG/$m" ] || die "missing $PROG/$m: build first (bci2000/README.md)"
done
pgrep -f "$PROG/.*(Source|ADC|Operator)" > /dev/null && die "BCI2000 is already running; close it first"
"$PY" "$REPO/tools/make_bci2000_parms.py" --check > /dev/null || die "bci2000/parms/ThaiCheckerboard.prm is out of date: run tools/make_bci2000_parms.py"

encode() {  # BCI2000 parameter encoding of a text, refusing characters that are not in the matrix
  "$PY" -c "import sys; sys.path.insert(0, '$REPO/tools'); from make_bci2000_parms import encode, load_characters
t = sys.argv[1]; bad = sorted({c for c in t if c not in load_characters()})
sys.exit('run_session: not in the matrix: ' + ' '.join(bad)) if bad else print(encode(t))" "$1" || exit 2
}

SP="DummySignalProcessing" APP="CheckerboardSpeller" SETTINGS=""
case "$MODE" in
  calibrate)
    [ -n "$TEXT" ] || die "calibrate needs --text (the calibration sentence)"
    ENCODED="$(encode "$TEXT")" || exit 2
    SETTINGS="Set parameter InterpretMode 2
Set parameter TextToSpell $ENCODED
Set parameter NumberOfSequences ${REPS:-5}" ;;
  online)
    [ -n "$MODEL" ] || die "online needs --model models/<id>_bci2000.prm (written by fit_from_recording)"
    MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")"
    [ -f "$MODEL" ] || die "no model at $MODEL"
    case "$MODEL" in *" "*) MODEL_LOAD="\"$MODEL\"" ;; *) MODEL_LOAD="$MODEL" ;; esac
    SP="CheckerboardSignalProcessing"
    if [ "$FREE" = 1 ]; then
      [ -z "$TEXT" ] || die "--free and --text exclude each other"
      MODE_LINES="Set parameter InterpretMode 1"
    else
      [ -n "$TEXT" ] || die "online needs --text (copy spelling, scored) or --free"
      ENCODED="$(encode "$TEXT")" || exit 2
      MODE_LINES="Set parameter InterpretMode 2
Set parameter TextToSpell $ENCODED"
    fi
    case "$CONDITION" in
      transformer) grep -q "LanguageModel= 1" "$MODEL" || die "$MODEL has no Transformer prior: refit with --lm-transformer-checkpoint, or use --condition uniform"
                   COND_LINES="" ;;
      uniform) COND_LINES="Set parameter LanguageModel 0" ;;
      fixed) COND_LINES="Set parameter StoppingRule 0" ;;
    esac
    # Bayesian stopping may run up to t_max flashes: 8 repetitions (192) cover the default t_max of 170.
    DEFAULT_REPS=8; [ "$CONDITION" = fixed ] && DEFAULT_REPS=5
    SETTINGS="Load parameterfile $MODEL_LOAD
$MODE_LINES
$COND_LINES
Set parameter NumberOfSequences ${REPS:-$DEFAULT_REPS}" ;;
  check)
    APP="DummyApplication" ;;
esac

if [ "$FAKE" = 1 ]; then
  SOURCE_ARGS="--TcpAddress=127.0.0.1:$PORT"
  echo "*** --fake: SYNTHETIC EEG from tools/fake_unicorn_bt.py. This is a rehearsal, not data. ***"
else
  SOURCE_ARGS=""
  ls /dev/cu.UN-* > /dev/null 2>&1 || die "no /dev/cu.UN-* port: pair the headset in System Settings > Bluetooth and switch it on"
fi

WORK="$(mktemp -d -t bci2000_session)"
cleanup() {
  pkill -f "$PROG/" 2> /dev/null || true
  [ -n "${FAKE_PID:-}" ] && { sleep 1; kill "$FAKE_PID" 2> /dev/null || true; }
}
trap cleanup EXIT INT TERM
if [ "$FAKE" = 1 ]; then
  lsof -i ":$PORT" > /dev/null 2>&1 && die "port $PORT is busy (an old fake headset? pkill -f fake_unicorn_bt)"
  "$PY" "$REPO/tools/fake_unicorn_bt.py" --port "$PORT" > "$WORK/fake_headset.log" 2>&1 &
  FAKE_PID=$!
  sleep 1
fi
mkdir -p "$DATA"
W="${WINDOW%x*}" H="${WINDOW#*x}"
WINDOW_LINES=""
if [ "$APP" = CheckerboardSpeller ]; then   # DummyApplication has no window parameters
  WINDOW_LINES="Set parameter WindowLeft 0
Set parameter WindowTop 40
Set parameter WindowWidth $W
Set parameter WindowHeight $H"
fi
DIR="$DATA/${SUBJECT}${SESSION}"
BEFORE="$(ls "$DIR"/*.dat 2> /dev/null || true)"

if [ "$MODE" = check ]; then
  RUN="Set config
Wait for Resting 30
Show window
Echo Check the traces in the Source window. The recording starts in 5 s: 20 s blinking, 20 s eyes closed, 20 s jaw clench.
Sleep 5
Start
Echo NOW: blink hard, about once a second (20 s)
Sleep 20
Echo NOW: close your eyes and relax (20 s)
Sleep 20
Echo NOW: clench your jaw, on and off (20 s)
Sleep 20
Stop
Wait for Suspended 10"
elif [ "$AUTO" = 1 ]; then
  RUN="Set config
Wait for Resting 30
Sleep 2
Start
Wait for Suspended 3600"
else
  RUN="Set config
Wait for Resting 30
Show window
Echo Configured. Check the signal in the Source window, then press Start in the Operator window.
Wait for Running
Wait for Suspended"
fi

cat > "$WORK/session.bat" << EOF
Change directory \$BCI2000LAUNCHDIR
Reset system
Startup system localhost --SystemLogFile=$WORK/system.log
Start executable UnicornSerialSource --local $SOURCE_ARGS
Start executable $SP --local
Start executable $APP --local
Wait for Connected 30
Load parameterfile "$SPELLER_PRM"
$SETTINGS
Set parameter DataDirectory "$DATA"
Set parameter SubjectName $SUBJECT
Set parameter SubjectSession $SESSION
$WINDOW_LINES
$RUN
Quit
EOF

echo "run_session: $MODE, subject $SUBJECT, session $SESSION$([ "$MODE" = online ] && echo ", condition $CONDITION")"
cd "$PROG"
env -u QT_QPA_PLATFORM ./BCI2000Shell "$WORK/session.bat" || echo "BCI2000 stopped with an error: see below."
cd "$REPO"

echo
if grep -q -E "Error" "$WORK/system.log" 2> /dev/null; then
  echo "--- BCI2000 errors ---"; grep -E "Error" "$WORK/system.log"
fi
# Only a file this run created counts: never report an earlier run's recording as this one's.
LATEST=""
while IFS= read -r f; do
  if ! printf '%s\n' "$BEFORE" | grep -qxF "$f"; then LATEST="$f"; break; fi
done < <(ls -t "$DIR"/*.dat 2> /dev/null || true)
[ -n "$LATEST" ] || { echo "No recording was written by this run (system log: $WORK/system.log)."; exit 1; }
echo "Recorded: $LATEST"
case "$MODE" in
  calibrate) echo "Next: $PY -m pipeline.fit_from_recording --recording \"$LATEST\" --id $SUBJECT \\
      --lm-transformer-checkpoint models/thai_gpt_checkpoint.pt$([ "$FAKE" = 1 ] && echo " --allow-demo")" ;;
  online) echo "Summary: $PY -m stopping.summarize_experiment --recording \"$LATEST\"" ;;
  check) echo "Checks: $PY tools/h13_montage_check.py \"$LATEST\"" ;;
esac
