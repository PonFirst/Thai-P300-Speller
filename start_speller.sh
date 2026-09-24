#!/usr/bin/env bash
#
# Launch the Thai P300 speller session control panel (live signal quality,
# live-inference display, and calibration/recording launcher).
#
# Usage:
#   ./start_speller.sh                          # real headset (Windows: the Unicorn LSL Interface;
#                                               Mac: python collection/unicorn_relay.py)
#   ./start_speller.sh --demo                   # no hardware: synthetic EEG + P300 responses
#   ./start_speller.sh --manifest path/to/session_manifest.json
#
# Any extra arguments (e.g. --manifest) are passed through to session_gui_qt.py.
# A relative --manifest is relative to the directory you run this from. Without
# --manifest the panel uses <repo>/session_manifest.json, or
# <repo>/demo_session_manifest.json with --demo.
#
# The previous Tkinter panel is still available as a fallback:
#   python collection/session_gui.py [--demo]
#
# NOTE: this only starts the LSL *consumer* side. The Unicorn LSL
# Interface (Windows) or collection/unicorn_relay.py (Mac) must already be
# streaming (on this machine -- see collection/README_ACQUISITION.md) before
# electrode dots turn green.
# This script itself is bash: on the Windows deployment machine, run it
# from Git Bash or WSL.

set -eo pipefail

ENV_NAME="p300speller"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI="$SCRIPT_DIR/collection/session_gui_qt.py"
MOCK="$SCRIPT_DIR/collection/mock_headset.py"

DEMO=0
GUI_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--demo" ]; then
        DEMO=1
    fi
    GUI_ARGS+=("$arg")
done

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found on PATH. Install Anaconda/Miniconda first." >&2
    exit 1
fi

# 'conda activate' is a shell function, not an executable, so it is not
# available in a non-interactive shell until the hook is sourced.
eval "$(conda shell.bash hook)"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "error: conda env '${ENV_NAME}' not found. Create it, then:" >&2
    echo "  conda activate ${ENV_NAME} && pip install -r requirements.txt" >&2
    exit 1
fi

conda activate "$ENV_NAME"

# Use the env's interpreter by absolute path, never a bare 'python': another
# virtualenv earlier on PATH would otherwise win even after 'conda activate'.
# The GUI launches its children (recorder, stimulus, inference service) with
# sys.executable, so they inherit this interpreter too.
PY="$(conda info --base)/envs/${ENV_NAME}/bin/python"
if [ ! -x "$PY" ]; then
    echo "error: interpreter for conda env '${ENV_NAME}' not found or not executable: $PY" >&2
    exit 1
fi

# The Qt panel needs PySide6 in this env. Fail here -- before the demo mock
# outlet starts -- with the remedy, instead of with a Python traceback.
if ! "$PY" -c "import PySide6" >/dev/null 2>&1; then
    echo "error: PySide6 is not installed for $PY (conda env '${ENV_NAME}')." >&2
    echo "  Install it:          $PY -m pip install PySide6" >&2
    echo "  Or use the Tk panel: $PY $SCRIPT_DIR/collection/session_gui.py [--demo]" >&2
    echo "                       (for --demo, start '$PY $MOCK' first)" >&2
    exit 1
fi

if [ "$DEMO" -eq 1 ]; then
    echo "=== DEMO MODE: synthetic EEG, no headset required ==="
    "$PY" "$MOCK" &
    MOCK_PID=$!
    # Kill the mock outlet however this script exits (normal quit, Ctrl+C,
    # error), so a stale fake 'EEG' stream cannot linger on the network and
    # get picked up by a later real session.
    trap 'kill "$MOCK_PID" 2>/dev/null || true' EXIT INT TERM
    sleep 2

    # NOT exec: exec would replace this shell and the cleanup trap above
    # would never run, orphaning the mock process.
    "$PY" "$GUI" "${GUI_ARGS[@]}"
else
    exec "$PY" "$GUI" ${1+"$@"}
fi
