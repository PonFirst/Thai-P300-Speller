# Running the speller on Windows (`windows` branch)

This branch adds Windows support to the BCI2000 path. The study itself runs on the Mac (branch `bci2000-port`); this is for running the same experiment on a Windows laptop.

**Status: written on a Mac, not yet built or run on Windows.** The first Windows user should treat the steps below as a checklist to verify, and report anything that fails.

What differs from the Mac:

| Piece | macOS | Windows |
|---|---|---|
| Headset port | `/dev/cu.UN-<serial>`, found automatically | a COM port you name: `-Port COM5` |
| Port code | `UnicornSerialSource/ByteSource.cpp` (POSIX) | `UnicornSerialSource/ByteSourceWin32.cpp` (Win32 COM port + Winsock) |
| Launcher | `bci2000/run_session.sh` | `bci2000/run_session.ps1` (PowerShell) |
| Python | `.../envs/p300speller/bin/python` | `...\envs\p300speller\python.exe` |

Everything else is the same code: the speller, the classifier, the stopping and the language model, the fit and all the analysis. Recordings made on Windows count as real headset data under the same rule as on the Mac (source `UnicornSerialADC`, no `TcpAddress`, no fake-headset gyroscope marker). They are directly comparable to Mac recordings.

## 1. Install

1. **Git** and **Miniconda** (or Anaconda). In a PowerShell window, run `conda init powershell` once, then open a new window.
2. **The repository:**
   ```powershell
   git clone <repository URL> P300Speller
   cd P300Speller
   git checkout windows
   ```
3. **The Python environment** (Python 3.11):
   ```powershell
   conda create -n p300speller python=3.11 -y
   & "$(conda info --base)\envs\p300speller\python.exe" -m pip install -r requirements.txt
   ```
4. **The language model checkpoint:** copy `models\thai_gpt_checkpoint.pt` from the project owner. It isn't in git.
5. **Let PowerShell run the launcher** (once): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 2. Build BCI2000 with this project's modules

1. Get the BCI2000 source and build tools (Visual Studio 2022 with "Desktop development with C++", and CMake) as described in BCI2000's own build instructions (bci2000.org wiki: *Programming Howto: Building and Customizing BCI2000*). The launcher expects the source at `%USERPROFILE%\BCI2000\trunk`; set the environment variable `BCI2000_ROOT` if it is elsewhere.
2. Link this folder into BCI2000 as its `custom` source folder (BCI2000 builds `src\custom` when it exists). In a **Command Prompt**:
   ```bat
   mklink /J "%USERPROFILE%\BCI2000\trunk\src\custom" "C:\path\to\P300Speller\bci2000"
   ```
3. Configure and build the BCI2000 core modules plus these three: **UnicornSerialSource**, **CheckerboardSpeller**, **CheckerboardSignalProcessing**. The launcher also needs `BCI2000Shell.exe`, `DummySignalProcessing.exe` and `DummyApplication.exe`. Everything lands in `%USERPROFILE%\BCI2000\trunk\prog\`.

## 3. Pair the headset and find its COM port

1. Switch the headset on. Windows Settings → **Bluetooth & devices** → **Add device** → Bluetooth → **UN-...**.
2. Bluetooth & devices → **More Bluetooth settings** (under "Related settings") → **COM Ports** tab. The headset has an **Outgoing** port, e.g. `COM5`. That is the port to use; the Incoming one won't work.
3. Display: set a fixed **60 Hz** refresh rate (Settings → System → Display → Advanced display), and keep the same setting for every session.

## 4. Run sessions

In PowerShell, from the repository folder:

```powershell
$PY = "$(conda info --base)\envs\p300speller\python.exe"
$env:PYTHONUTF8 = "1"        # Thai text in Python's console output

.\bci2000\run_session.ps1 check     -Subject S01 -Port COM5
.\bci2000\run_session.ps1 calibrate -Subject S01 -Port COM5 -Text "<calibration sentence>"
.\bci2000\run_session.ps1 online    -Subject S01 -Port COM5 -Model models\S01_bci2000.prm -Text "<sentence>" -Condition fixed
.\bci2000\run_session.ps1 online    -Subject S01 -Port COM5 -Model models\S01_bci2000.prm -Text "<sentence>" -Condition transformer
```

- Each run prints the next command to use: the fit after calibration, the summary after an online run, the montage check after a check run.
- Recordings, `.applog` and `.system.log` files go to `collection\recordings\bci2000\<Subject><Session>\`.
- The analysis commands are the same as on the Mac, with `& $PY` in place of `$PY`:
  ```powershell
  & $PY tools\h13_montage_check.py collection\recordings\bci2000\S01001\S01S001R01.dat
  & $PY -m stopping.summarize_experiment --recording <run>.dat --html <run>_report.html
  ```

**Without the headset:** add `-Fake` instead of `-Port` to any run. It uses `tools\fake_unicorn_bt.py`, and the data marks itself synthetic. That's the first thing to try after building.

**The colour-coded signal check** (the Python panel) works too: `& $PY collection\unicorn_relay.py --port COM5` in one window, then the panel. Close both before starting BCI2000: only one program can hold the COM port.

## 5. Things to verify on the first Windows run

1. `-Fake` runs end to end: check → calibrate → fit → online → summary.
2. The headset streams (`check -Port COM5`): the run's `.system.log` reports packets received and few samples lost.
3. The headset's first data arrives within `StartupTimeout` (10 s). On the Mac it takes 3.5–4 s after the port opens.
4. Thai characters render correctly in the speller window, including tone marks and lone combining marks. BCI2000 draws with the system's fonts.
5. A run started while the headset is off fails with a clear message, and the headset works again after switching it off and on.

Report the results, with the terminal output and the `.system.log`, so this page can drop its "not yet run" status.
