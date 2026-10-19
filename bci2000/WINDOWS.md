# The Windows package (`windows` branch)

Windows users don't build anything. They download one zip, unzip it to `C:\`, pair the headset and double-click `ThaiSpeller.cmd`. Their guide is [`START_HERE.txt`](../START_HERE.txt), which ships at the top of the package.

The study's own sessions run on the Mac (branch `bci2000-port`). This branch is for running the same experiment on Windows laptops.

## What's in the zip

| Folder | Contents |
|---|---|
| `ThaiSpeller.cmd`, `START_HERE.txt` | the starter and the user's guide |
| `bci2000\prog\` | BCI2000 (revision 9529, as on the Mac) with `UnicornSerialSource`, `CheckerboardSpeller`, `CheckerboardSignalProcessing`, plus Qt and the Visual C++ runtime |
| `python\` | a private Python 3.11 with `bci2000\windows\requirements.txt` (no PyTorch) |
| `models\thai_gpt.lmbin` | the Transformer prior, read by the speller in C++ and by Python in NumPy |
| everything else | this repository at the commit in `VERSION.txt` |

Differences from the Mac, all inside the package:
- **Headset port:** `UnicornSerialSource` reads a COM port (`ByteSourceWin32.cpp`). `session_helper.py find-headset` picks the paired headset's outgoing Bluetooth port, so nobody types a port.
- **Launcher:** `bci2000\run_session.ps1` (the counterpart of `run_session.sh`), driven by the menu `bci2000\windows\menu.ps1`. The calibration and spelling sentences come from `bci2000\windows\sentences.json`.
- **Transformer without PyTorch:** the fit copies `models\thai_gpt.lmbin` instead of exporting a `.pt`, and replays read it through `stopping/thai_gpt_numpy.py`. It equals the PyTorch model to about 4e-7 (`tests/test_thai_gpt_numpy.py`).

Recordings are the same `.dat` format, from the same modules and code, so they are directly comparable to the Mac's, and they count as real headset data under the same rule.

## Building a package

The GitHub Actions workflow `.github/workflows/windows-package.yml` builds it on GitHub's Windows machines.

**Once:** add your bci2000.org login as repository secrets (Settings → Secrets and variables → Actions): `BCI2000_SVN_USERNAME` and `BCI2000_SVN_PASSWORD`.

**Each build:**
1. Push to the `windows` branch. After merging Mac changes: `git checkout windows && git merge bci2000-port && git push`.
2. The run (Actions tab, about an hour the first time; the BCI2000 source is cached afterwards):
   - checks out BCI2000 and builds it with this project's modules;
   - assembles the package;
   - runs the Python tests with the package's own Python;
   - uploads the zip as the artifact `ThaiSpeller-Windows`;
   - rehearses a whole session (check → calibration × 2 → fit → both spelling runs → reports) through the menu with the fake headset in Windows PowerShell, as users run it. Its files are kept as the artifact `rehearsal-logs`.
3. To publish a release: `git tag win-v1 && git push origin win-v1`. The zip then appears on the repository's Releases page, which is the link to give users.

**Licensing:** the zip contains BCI2000, which is GPL software, and our modules built into it. Whoever receives the zip must be able to get the full source, so publish releases only while this repository is public (or give recipients access to it). `NOTICE.txt` in the package lists the components.

## Status

Written on the Mac. The Windows serial code and both PowerShell scripts parse and match the Mac logic, but nothing has run on Windows until the first workflow run. That run's rehearsal is the first real test.

Still to confirm on a real Windows laptop with the headset:
1. The headset is found automatically and streams. Its first data arrives within `StartupTimeout` (10 s); on the Mac it takes 3.5–4 s.
2. The check run passes (H13, H3), and `.system.log` shows little packet loss.
3. Thai renders correctly in the speller window.
4. Display timing (TESTING.md H7), before pooling Windows data with Mac data.

## Developing on Windows without the package

`run_session.ps1` and `menu.ps1` also work from a plain clone. They fall back to the `p300speller` conda environment and to a BCI2000 build at `%USERPROFILE%\BCI2000\trunk` (or `BCI2000_ROOT`), with this folder linked as `src\custom` (`mklink /J`). The workflow file shows the exact build steps.
