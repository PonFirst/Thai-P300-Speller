#!/usr/bin/env bash
#
# One-time environment setup for the Thai P300 speller on a fresh Windows
# machine. Run from Git Bash or WSL (this is a bash script, like
# start_speller.sh -- it will not run in PowerShell or cmd.exe directly).
#
# Creates the p300speller conda environment, installs every dependency from
# requirements.txt, and verifies the packages most likely to have
# Windows-specific install problems (PySide6, PsychoPy, pylsl) actually
# import before you trust the environment.
#
# This script does NOT touch the headset, the Unicorn Suite, or the
# Bluetooth pairing -- see WINDOWS_SETUP.md and
# collection/README_ACQUISITION.md for that. Run this first; do the
# headset/software pairing separately, in any order relative to this.
#
# Usage:
#   bash setup_windows.sh              # create the env (reuses it if it already exists) + install + verify
#   bash setup_windows.sh --recreate   # delete and recreate the env from scratch
#   bash setup_windows.sh --test       # also run the full test suite as a final check

set -eo pipefail

ENV_NAME="p300speller"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RECREATE=0
RUN_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        --test) RUN_TESTS=1 ;;
        *) echo "unknown argument: $arg (expected --recreate and/or --test)" >&2; exit 2 ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found on PATH." >&2
    echo "  Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
    echo "  Then close and reopen Git Bash/WSL so 'conda' is on PATH, and re-run this script." >&2
    exit 1
fi

# conda create/env list/env remove are plain 'conda' subcommands -- no
# 'conda activate' needed for any of them.
ENV_EXISTS=0
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    ENV_EXISTS=1
fi

if [ "$RECREATE" -eq 1 ] && [ "$ENV_EXISTS" -eq 1 ]; then
    echo "=== Removing existing '${ENV_NAME}' environment (--recreate) ==="
    conda env remove -n "$ENV_NAME" -y
    ENV_EXISTS=0
fi

if [ "$ENV_EXISTS" -eq 0 ]; then
    echo "=== Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION}) ==="
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
else
    echo "=== Environment '${ENV_NAME}' already exists -- reusing it (pass --recreate to start over) ==="
fi

# Resolve the env's interpreter as a hand-built absolute path -- the same
# thing start_speller.sh does, and for the same reason: on the Mac this repo
# is developed on, a pipenv virtualenv earlier on PATH shadows 'python' even
# after 'conda activate', so activating and trusting a bare 'python' (or
# 'conda run', which was tried here and hit the exact same shadowing) is not
# safe. Windows conda puts python.exe at the env root, not in bin/, so check
# both layouts rather than assuming one.
CONDA_BASE="$(conda info --base)"
PY_UNIX="$CONDA_BASE/envs/${ENV_NAME}/bin/python"
PY_WIN="$CONDA_BASE/envs/${ENV_NAME}/python.exe"
if [ -x "$PY_UNIX" ]; then
    PY="$PY_UNIX"
elif [ -f "$PY_WIN" ]; then
    PY="$PY_WIN"
else
    echo "error: could not find the interpreter for conda env '${ENV_NAME}' at either:" >&2
    echo "  $PY_UNIX" >&2
    echo "  $PY_WIN" >&2
    exit 1
fi
echo "Using interpreter: $PY"

echo ""
echo "=== Installing dependencies from requirements.txt ==="
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Verifying the packages most likely to have Windows-specific issues ==="
FAILED=0
for pkg in numpy scipy pandas sklearn mne moabb pyriemann matplotlib joblib psychopy pylsl PySide6 pytest; do
    if "$PY" -c "import $pkg" >/dev/null 2>&1; then
        echo "  OK    $pkg"
    else
        echo "  FAIL  $pkg"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "" >&2
    echo "error: one or more packages failed to import -- see FAIL lines above." >&2
    echo "PsychoPy and PySide6 are the most common culprits on Windows (a missing" >&2
    echo "Visual C++ Redistributable, or a conflicting Qt install elsewhere on the" >&2
    echo "machine). Fix the specific import error before continuing -- do not move" >&2
    echo "on to real data collection with a partially-working environment." >&2
    exit 1
fi

echo ""
echo "=== Environment '${ENV_NAME}' is ready ==="
echo "PY=\"$PY\""

if [ "$RUN_TESTS" -eq 1 ]; then
    echo ""
    echo "=== Running the test suite (QT_QPA_PLATFORM=offscreen) ==="
    echo "Expected: everything passes; 3 tests self-skip (see CLAUDE.md)."
    QT_QPA_PLATFORM=offscreen "$PY" -m pytest "$SCRIPT_DIR/tests" -q
fi

echo ""
echo "Next steps (see WINDOWS_SETUP.md for the full walkthrough):"
echo "  1. Pair the headset: install Unicorn Suite, plug in the Bluetooth"
echo "     dongle, pair in the 'My Unicorn' tab, start the Unicorn LSL"
echo "     Interface and confirm it says 'Streaming'."
echo "  2. bash start_speller.sh --demo   (no headset needed -- confirms the panel itself works)"
echo "  3. bash start_speller.sh          (real headset)"
