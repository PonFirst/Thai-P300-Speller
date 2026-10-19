# Thai P300 Speller

A Thai-language P300 speller in [BCI2000](https://www.bci2000.org): a checkerboard matrix of 72 Thai characters, recorded with the g.tec Unicorn Hybrid Black (8 channels, 250 Hz). It adds Bayesian dynamic stopping, which ends a character's flashes once the speller is confident, and a Thai character-level Transformer language model as the prior.

All EEG is recorded by BCI2000. Python fits the classifier between runs and analyses the recordings afterwards.

## Which branch

| Branch | What it is |
|---|---|
| **`windows`** | **Use this one.** Everything below, on Windows and macOS: the BCI2000 modules, the session tools, the language model file, and the Windows package. |
| `bci2000-port` | The macOS development branch that `windows` is built on. |
| `main` | The earlier, Python-only version of the project (no BCI2000). |

## Running it on Windows (no installation)

The Windows package contains BCI2000, Python and the model, ready to run.

### Set up (once, about 5 minutes)

1. **Download `ThaiSpeller-Windows.zip`**, either from this repository's **Releases** page, or from **Actions** → the latest successful **Windows package** run → **Artifacts**.
2. **Unblock it:** right-click the zip → **Properties** → tick **Unblock** at the bottom → **OK**. If there's no Unblock box, skip this.
3. **Unzip it to `C:\`:** right-click → **Extract All...** → change the folder to `C:\` → **Extract**. This makes `C:\ThaiSpeller`.
4. **Pair the headset:** switch it on, then Settings → **Bluetooth & devices** → **Add device** → **Bluetooth** → **UN-...**.

### Each session

1. **Laptop and room:** unplug the charger (it adds 50 Hz noise to the EEG), and set the display to a fixed 60 Hz (Settings → System → Display → Advanced display). Use a quiet room.
2. **Headset on:** gel in all 8 electrodes, and both sticker electrodes on the bone behind the ears with their cables clipped in.
3. **Start:** double-click `C:\ThaiSpeller\ThaiSpeller.cmd`.
   - "Windows protected your PC": **More info** → **Run anyway**.
   - Firewall asks about BCI2000 or Python: **Allow**. They only talk to each other on this laptop.
4. **Answer the questions:** `Y` for the real headset, the participant code (letters and digits, e.g. `S01`), and the session number.
5. **Run the steps in order** by typing their number (see [The session](#the-session)). The menu marks finished steps **done**.

`START_HERE.txt` in the package has the same steps.

## Running it on macOS (build from source)

### Set up (once)

You need a free [bci2000.org](https://www.bci2000.org) account for the BCI2000 source, plus Xcode Command Line Tools, CMake, Qt 6 (from the Qt online installer), Miniconda and Subversion.

```bash
# BCI2000, at the revision this project is tested with
svn checkout -r 9529 https://bci2000.org/svn/trunk ~/BCI2000/trunk

# This repository, with its modules linked into BCI2000
git clone -b windows https://github.com/PonFirst/Thai-P300-Speller.git
cd Thai-P300-Speller
ln -s "$PWD/bci2000" ~/BCI2000/trunk/src/custom

# Build (the first build takes a while)
cd ~/BCI2000/trunk/build
cmake . -DBUILD_MODULES=core -DBUILD_CUSTOM=ON -DCMAKE_PREFIX_PATH="$HOME/Qt/<version>/macos"
make -j8
cd -

# Python
conda create -n p300speller python=3.11 -y
"$(conda info --base)/envs/p300speller/bin/python" -m pip install -r bci2000/windows/requirements.txt
```

### Each session

In the repository folder, set these once per terminal:

```bash
PY="$(conda info --base)/envs/p300speller/bin/python"
SUBJ=S01            # participant code: letters and digits only
SESS=001            # session number
D=collection/recordings/bci2000/${SUBJ}${SESS}
```

Pair the headset in System Settings → Bluetooth. `ls /dev/cu.UN-*` must list it. Then:

| Step | Command |
|---|---|
| Check run | `bci2000/run_session.sh check --subject $SUBJ --session $SESS` <br> `$PY tools/h13_montage_check.py $D/${SUBJ}S${SESS}R01.dat` |
| Calibration 1 | `bci2000/run_session.sh calibrate --subject $SUBJ --session $SESS --text "นกฮูกเกาะกิ่งไม้"` |
| Calibration 2 | `bci2000/run_session.sh calibrate --subject $SUBJ --session $SESS --text "ฟังเสียงฝนตกทั้งคืน"` |
| Fit | `$PY -m pipeline.fit_from_recording --id $SUBJ --recording $D/${SUBJ}S${SESS}R02.dat $D/${SUBJ}S${SESS}R03.dat --manifest models/${SUBJ}_manifest.json --lm-transformer-checkpoint models/thai_gpt.lmbin` |
| Spelling, all flashes | `bci2000/run_session.sh online --subject $SUBJ --session $SESS --model models/${SUBJ}_bci2000.prm --condition fixed --text "วันนี้ผมไปกินข้าวที่ตลาด"` |
| Spelling, dynamic stopping | the same with `--condition transformer` |
| Reports | `$PY -m stopping.summarize_experiment --recording <run>.dat --html <run>_report.html` |

If a run is repeated, BCI2000 numbers the files on (R04, R05, …); use the file name each run prints after **Recorded:**.

## The session

| # | Step | Time | What happens |
|---|---|---|---|
| 1 | Look at the signal | as needed | Live EEG in BCI2000's Source window; nothing is recorded. Adjust electrodes until all 8 traces are calm. |
| 2 | Check run | 1.5 min | Cued blinking, eyes closed and jaw clenching. Continue only if all four result lines say **PASS** (channel order and signal range). |
| 3 | Calibration, part 1 | 6 min | Spells *นกฮูกเกาะกิ่งไม้* (16 characters). |
| 4 | Calibration, part 2 | 7 min | Spells *ฟังเสียงฝนตกทั้งคืน* (19 characters). Rest a few minutes first. |
| 5 | Fit | 1–2 min | Trains the classifier on both calibration parts. Continue if the out-of-fold AUC is about 0.70 or more. |
| 6 | Spelling, all flashes | 8.5 min | *วันนี้ผมไปกินข้าวที่ตลาด* with all 120 flashes per character. |
| 7 | Spelling, dynamic stopping | 5–8.5 min | The same sentence. Stops when one character reaches 90 % probability, starting from the Transformer's prediction. |
| 8 | Reports | 2 min | Accuracy with a 95 % confidence interval, information transfer rate, the language model's effect, and a replay over stopping thresholds. |

**Participant instructions** for steps 3, 4, 6 and 7: press Start in the Operator window, look at the highlighted character, silently count how often it flashes, and blink only between characters.

**Total:** about an hour, including fitting the headset.

## What is saved

Everything for one session is in `collection/recordings/bci2000/<code><session>/`, plus the participant's model in `models/`:

| File | Contents |
|---|---|
| `*.dat` | The BCI2000 recording: EEG in µV at 250 Hz, lost Bluetooth samples, every flash and the characters it lit, the cued character, classifier scores, selections, stop reasons, and every setting of the run |
| `*.applog` | The language model's prior for every character |
| `*.system.log` | BCI2000's messages and the headset's packet-loss report |
| `models/<code>_fit_report.json` | Cross-validation AUC per fold, calibration, the input files with checksums, software versions |
| `report_*.html`, `replay_*.csv` | Session results and the threshold sweep |

## Trying it without the headset

- **Windows:** start `ThaiSpeller.cmd` and type `R` instead of `Y`.
- **macOS:** add `--fake` to any `run_session.sh` command.

A fake headset sends synthetic EEG through the same modules. Its files are marked synthetic, kept separate from real recordings, and its accuracy is at chance.

## Troubleshooting

| Message or symptom | Fix |
|---|---|
| No paired headset found / no `/dev/cu.UN-*` | Switch the headset on and pair it. |
| The headset sent nothing / no data from the headset | Switch the headset off for 10 seconds, on again, and repeat the step. If it persists, remove the headset from Bluetooth and pair it again. |
| Every channel shows the same huge noise | The ground or reference sticker is loose, or the charger is plugged in. |
| BCI2000 is already running | Close the open Operator window first. |
| Not in the matrix | The text contains a character the speller doesn't have, for example a space. |

## How it works

- **Classifier:** xDAWN spatial filters → covariance → tangent space → shrinkage LDA, scoring every flash, with Platt scaling to turn scores into probabilities. It is trained per participant from the calibration runs. Online scoring runs inside BCI2000 (C++) and matches the Python fit.
- **Dynamic stopping:** a Bayesian posterior over the 72 characters, updated after every flash. A character is selected when its probability reaches 0.9 (after at least 6 flashes), or at 120 flashes.
- **Language model:** a character-level Transformer (6 layers, 5.6 M parameters) trained on Thai Wikipedia. It gives the prior for each character from the text spelled so far.
- **BCI2000 modules** (`bci2000/`): `UnicornSerialSource` (the headset over Bluetooth), `CheckerboardSpeller` (matrix, stopping, language model), `CheckerboardSignalProcessing` (the classifier).

`CLAUDE.md`, `bci2000/README.md` and `bci2000/WINDOWS.md` have the technical details.
