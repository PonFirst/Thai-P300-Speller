#!/usr/bin/env bash
#
# Launch the Thai P300 speller session control panel (live signal quality
# + calibration/recording launcher).
#
# Usage:
#   ./start_speller.sh           # real headset (requires the Unicorn LSL Interface)
#   ./start_speller.sh --demo    # no hardware: synthetic EEG + P300 responses
#
# Any extra arguments are passed through to session_gui.py.
#
# NOTE: this only starts the LSL *consumer* side. The Unicorn LSL
# Interface must already be streaming from the Windows VM / Linux box
# before electrode dots turn green -- see collection/README_ACQUISITION.md.

set -eo pipefail

ENV_NAME="p300speller"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEMO=0
GUI_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--demo" ]; then
        DEMO=1
        GUI_ARGS+=("--demo")
    else
        GUI_ARGS+=("$arg")
    fi
done

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found on PATH. Install Anaconda/Miniconda first." >&2
    exit 1
fi

# 'conda activate' is a shell function, not an executable, so it is not
# available in a non-interactive shell until the hook is sourced. Without
# this the script fails when run directly as ./start_speller.sh.
eval "$(conda shell.bash hook)"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "error: conda env '${ENV_NAME}' not found. Create it, then:" >&2
    echo "  conda activate ${ENV_NAME} && pip install -r requirements.txt" >&2
    exit 1
fi

conda activate "$ENV_NAME"

cd "$SCRIPT_DIR/collection"

if [ "$DEMO" -eq 1 ]; then
    echo "=== DEMO MODE: synthetic EEG, no headset required ==="
    python mock_headset.py &
    MOCK_PID=$!
    # Kill the mock outlet however this script exits (normal quit, Ctrl+C,
    # error), so a stale fake 'EEG' stream cannot linger on the network and
    # get picked up by a later real session.
    trap 'kill "$MOCK_PID" 2>/dev/null || true' EXIT INT TERM
    sleep 2

    # NOT exec: exec would replace this shell and the cleanup trap above
    # would never run, orphaning the mock process.
    python session_gui.py "${GUI_ARGS[@]}"
else
    exec python session_gui.py ${1+"$@"}
fi
