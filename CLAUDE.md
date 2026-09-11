# Thai P300 Speller — Context for Claude Code

## What this is

PhD application portfolio work (targeting CMU ECE/BME): a Thai-language P300 visual speller extending Pananookooln et al. 2022 (AIT Brain Lab). Checkerboard paradigm, 72 Thai characters, 8-channel EEG at 250 Hz (g.tec Unicorn Hybrid Black).

- **The classifier** (xDAWN + RegLDA, `classifiers/`) is validated and not the contribution.
- **The contribution** is Bayesian dynamic stopping (`stopping/`): a fixed confidence threshold plus a Thai character-level LM prior.
  - The LM is an n-gram today.
  - The owner will build a character-level Transformer later. It will be a new `language_model` type in `pipeline/manifest.py` plus a builder in `pipeline/language_model.py`.
- **Boundary-aware thresholding is descoped.** The paper's novelty angle is **not decided**; don't assume one.

## Hard constraints — do not relitigate

- **Hardware path.**
  - The dev machine is an Apple M3 Mac. BrainFlow's Unicorn class is compiled out on macOS, and g.tec's tools are Windows-only.
  - Acquisition runs through the vendor LSL bridge on a Windows-on-ARM VM or a Linux box; everything in this repo is a pure LSL consumer.
  - Never suggest `pip install brainflow`.
- **The owner implements the core stopping math.**
  - `BayesianStoppingController.reset()`, `.llr()`, `.update()` stay `raise NotImplementedError` until the owner fills them in.
  - Never implement, rewrite, or "fix" them, not even to make tests pass or inside test doubles. Mocks return scripted literal values.
  - Everything around them is fair game.
- **Every recording logs full per-flash metadata.** Never regress this.
- **Don't modify the stimulus or the classifier without an explicit ask.**
  - `collection/checkerboard_stimulus.py` was out of scope in the last round.
  - `classifiers/` is validated: fit it, don't change it.

## Environment

- A pipenv virtualenv (`storefront`) sits earlier on `PATH` than conda, so **`python` is the wrong interpreter even after `conda activate p300speller`**. Always call the env's interpreter by path: `PY="$(conda info --base)/envs/p300speller/bin/python"`.
- `start_speller.sh` does this for you, and the GUI launches its child processes with `sys.executable`.
- GUI tests need `QT_QPA_PLATFORM=offscreen`.

## Contracts

`docs/INTERFACES.md` is the reference for how the pieces talk to each other. Code in `pipeline/` defines the contracts; import it, never redefine it.

| What | Where |
|---|---|
| Stream names/types, channel order, 250 Hz, µV thresholds, the dejitter switch | `pipeline/constants.py` |
| `P300Inference` events (service → GUI), strictly validated on both ends | `pipeline/events.py` |
| `session_manifest.json`: which model files a run uses (sha256-verified, fingerprinted) | `pipeline/manifest.py` |
| Quality levels + quality gate (shared by GUI map and service) | `pipeline/quality.py` |
| Epoch cutting (shared by fit script and service: train/serve parity) | `pipeline/epoching.py` |
| Marker JSON (`trial_start` / `flash` / `trial_end`) | `collection/checkerboard_stimulus.py` |
| Recorder CSV + gap rows + `<csv>.log.jsonl` + `<csv>.integrity.json` | `collection/record_session.py` docstring |

- **Resolution:** all consumers find streams **by type**, and every tool takes `--eeg-type` / `--marker-type` / `--inference-type`.
- **`P300Inference` stream type:** it is deliberately not `Markers`.

## Repo map

```
start_speller.sh                 ./start_speller.sh [--demo] [--manifest PATH] -> Qt panel
collection/
  session_gui_qt.py              PySide6 panel: traces, scalp map, target entry, settings, composed text,
                                 confidence gauge; launches service -> recorder -> stimulus
  gui_logic.py                   all panel logic without Qt (tested): settings, paths, connection state,
                                 inference display model, SessionOrchestrator
  board_qt.py, board_schedule.py in-window stimulus board (the panel's default) and its timeline;
                                 same markers and trial structure as checkerboard_stimulus.py
  session_gui.py                 LEGACY Tkinter panel, kept as a fallback
  record_session.py              hardened LSL recorder (backoff, gap rows, integrity checks, JSON logs)
  checkerboard_stimulus.py       PsychoPy matrix, cue phase, JSON markers at flip time
  flash_sequence.py              pure checkerboard grouping logic
  mock_headset.py                synthetic EEG outlet: fault flags, P300 on target flashes, fixed sample grid
  README_ACQUISITION.md          machine topology, startup order, recorder output and exit codes
  checkerboard.py, checkerboard_experiment.py, test_sync.py   LEGACY, pre-contract; don't use
pipeline/
  constants.py quality.py events.py manifest.py epoching.py   shared contracts
  inference_service.py           live inference: LSL in, P300Inference out (InferenceEngine is pure)
  fit_from_recording.py          calibration CSV -> classifier checkpoint + Platt calibration + manifest
  classifier_io.py               checkpoint save/load, stream compatibility, channel-label policy
  language_model.py              LM factory (uniform | ngram), prior validated at the boundary
stopping/
  bayesian_stopping.py           log_normalize (done) + controller (3 owner-implemented stubs)
  calibrate.py lm_prior.py replay.py   Platt/ECE, ThaiNgramLM, replay + Wolpaw ITR
  validate_offline.py            code validation on a FABRICATED grid, not a research result
  flash_selector.py              info-gain ordering stubs, out of scope
tools/
  fake_inference_publisher.py    scripted P300Inference stream for GUI QA
  e2e_demo_smoke.sh              no-hardware end to end: record -> fit -> service ready (opens a window)
classifiers/                     benchmark_all.get_pipelines() = single source of the 5 pipelines
config.py, data_loader.py        MOABB/BNCI EDA path (volts!)
tests/                           see TESTING.md; test_fault_injection.py = cross-component scenarios
docs/INTERFACES.md, TESTING.md
```

## Running

```bash
./start_speller.sh --demo                       # no hardware: mock EEG + panel; recordings prefixed demo_
./start_speller.sh                              # real headset; start the Unicorn LSL Interface FIRST
$PY collection/mock_headset.py --flat P3 --noisy C4 --railed O1       # exercise quality colours
$PY -m pipeline.fit_from_recording --recording collection/recordings/<file>.csv --id subject_001
$PY -m pipeline.inference_service --manifest session_manifest.json    # normally launched by the panel
$PY tools/fake_inference_publisher.py --include-malformed             # drive the composed-text panel
QT_QPA_PLATFORM=offscreen $PY -m pytest tests/ -q
```

- **Expected tests:** all pass except the 4 intentional stubs in `tests/test_bayesian_stopping.py`.
- **Output locations:** recordings go to `collection/recordings/` and models to `models/` (both gitignored). Demo artifacts are always `demo_`-prefixed.

## Known gotchas — don't silently reintroduce

1. **Units split by subsystem.**
   - Collection, pipeline, and GUI work in **µV**.
   - The MOABB path (`data_loader.py`, `config.PTP_REJECT_THRESHOLD_V`) works in **volts**.
   - Never mix the two sets of thresholds. An earlier unit bug silently rejected ~100 % of epochs.
2. **LSL never delivers samples pushed before a consumer subscribed.**
   - The stimulus waits for a marker consumer (15 s timeout), plus 2 s for the rest, before its first trial (`wait_for_marker_consumers`). Without this, the first `trial_start` of every run was lost.
   - If no consumer appears, it warns and starts anyway. The recorder still flags any trial without a `trial_start` (`trials_without_trial_start`).
   - The recorder records EEG first and opens the marker inlet the moment it resolves, which fixed the older 2.7 s marker loss.
3. **An unopened LSL inlet buffers nothing.** Call `open_stream()` right after creating an inlet (measured: 0/10 markers received without it).
4. **LSL doesn't replay samples to late subscribers.** The service re-sends `ready` every 2 s, and the GUI trusts the stream desc's fingerprint.
5. **Timestamps are recorded as delivered: clock sync, no dejitter.**
   - Measured: dejitter leaves clean outlets unchanged but silently shifts up to 19 ms to hide out-of-order stamps.
   - If the real device proves to need it, flip `EEG_INLET_DEJITTER` in `pipeline/constants.py`; the recorder and service follow it together.
6. **Flat channels:** peak-to-peak over the 1 s window alone kept a failed contact green for about a second. A channel is now also red when any 100 ms segment is below 1 µV. The thresholds themselves are unchanged.
7. **Matrix orientation is unresolved.** The stimulus defaults to 8×9 (odd columns, adjacency-safe). 9×8 is NOT safe under flat-index parity without reordering the characters.
8. **`characters_th.json` is `"verified": false`.** Confirm it against `th_offline_cb.py` before real collection.
9. **The montage is unverified.** `EEG_CHANNELS` = Fz, F3, C4, Cz, Pz, P3, O2, O1 is not g.tec's default layout.
   - The real device's channel labels are unknown.
   - The service warns on generic labels and refuses when the electrode names differ.
   - See TESTING.md H1/H2.
10. **Thai fonts.**
    - The GUI uses Sarabun, falling back to Thonburi; its tone-mark stacking is verified by an automated test.
    - The PsychoPy stimulus still falls back from the missing `THSarabunNew.ttf` and is unverified.
11. **BNCI2014-008/009 expose no row/column identity.** Report flash-level metrics only there.
12. **Baseline correction** (`BASELINE_WINDOW`) fixed a sub-3 µV Fz artifact on the MOABB path. Don't remove it.
13. **CV is grouped by subject, not by trial, on BNCI.** Fit-script calibration is out-of-fold, grouped by `trial_id`.
14. **~5:1 class imbalance.** Report balanced accuracy / AUC.
15. **FgMDM and TS+RegLDA have identical AUC — expected.** pyriemann 0.12's FGDA runs the same shrinkage LDA on the same features.
16. **Mock-data AUC is low (~0.6) by design** (a 6 µV P300 in ~9 µV single-epoch noise). It isn't a classifier result.
17. **Stale references.** `character_decoder.py` is cited but doesn't exist; `setup.py` names `day2_moabb_setup.py`; `validate_offline.py` prints `theta_base`.
18. **In-window board timing is estimated.**
    - Qt on macOS has no presentation timestamp, so the board stamps markers when it requests a repaint.
    - Measured: on schedule within 7 ms; the lag to the screen is unmeasured.
    - Use Board = "Separate window (precise timing)" for real data until TESTING.md H7 passes.

## Results so far (flash-level, subjects 1–5, GroupKFold by subject)

From `results/all_classifiers_results.csv` (AUC / balanced accuracy):

| Pipeline | BNCI2014_008 | BNCI2014_009 |
|---|---|---|
| xDAWNcov+FgMDM | 0.703 / 0.627 | 0.824 / 0.698 |
| xDAWNcov+TS+RegLDA | 0.703 / 0.548 | 0.824 / 0.630 |
| xDAWNcov+MDM | 0.646 / 0.574 | 0.835 / 0.709 |
| xDAWNcov+MeanField | 0.640 / 0.560 | 0.835 / 0.714 |
| xDAWN+LR | 0.679 / 0.631 | 0.718 / 0.621 |

- **The best pipeline differs by dataset.** Report that; don't average it away.
- **Unreconciled earlier run.** A `benchmark_all.py` run reported FgMDM at 0.769 / 0.694 on 008. Reconcile the configurations before citing either.
- **Subject 4 was the worst fold everywhere.** Treat it as a finding.

## Conventions

- **Filenames:** no day-number prefixes.
- **Pipelines:** new ones go in `get_pipelines()`.
- **Shared constants:** import them; don't redeclare.
- **Pure logic apart from I/O:** keep it separate, and test it without hardware.
  - Examples: `flash_sequence`, `gui_logic`, `InferenceEngine`, `mock_headset`'s pure functions.
- **Threshold tests:** check every channel across several seeds, with margin. Single-seed tests have missed two boundary bugs here.
- **Real-LSL tests:** use randomized stream types, never the production ones.
- **Tests never write into the repo.**
- **Synthetic data must identify itself:** `MockUnicornEEG`, the `demo_` prefix, the GUI banner.
- **Sanity checks:** check real output against plausible ranges before trusting it.
- **Library behaviour:** check it against source or measurement before relying on it. That habit found gotchas 1, 3, 5 and 15.
