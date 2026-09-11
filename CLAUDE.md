# Thai P300 Speller — Context for Claude Code

## What this is

PhD application portfolio work (targeting CMU ECE/BME). A Thai-language P300
visual speller extending Pananookooln et al. 2022 (AIT Brain Lab): checkerboard
paradigm, 72 Thai characters, 8-channel EEG at 250 Hz (g.tec Unicorn Hybrid Black).

- **Classifier** (xDAWN-family, `classifiers/`) is built and validated; not
  the contribution.
- **The contribution** is a Bayesian dynamic-stopping module (`stopping/`):
  fixed confidence threshold + a Thai character-level LM prior.
- **Boundary-aware thresholding is descoped.** Don't build boundary-probability
  or word-segmentation logic unless the user reopens it.
- The novelty angle for the paper is **not yet decided** — don't assume one.

## Hard constraints — do not relitigate

- **Hardware path.** Dev machine is an Apple M3 Mac. BrainFlow's Unicorn board
  class is compiled out on macOS (`unicorn_board.h` is wrapped in
  `#if defined __linux__ || defined _WIN32`), and g.tec's Unicorn Suite /
  LSL Interface is Windows-only. Acquisition = vendor LSL Interface on a
  Windows-on-ARM VM (USB dongle passed through) or a Linux box. Everything in
  this repo is a pure LSL consumer. Never suggest `pip install brainflow`.
- **The user implements the core stopping math.** `BayesianStoppingController.reset()`,
  `.llr()`, `.update()` in `stopping/bayesian_stopping.py` must stay
  `raise NotImplementedError` — deliberate, for interview mastery. Do not fill
  them in, even to "make tests pass". Everything around them is fair game.
- **Every recording logs full per-flash metadata.** The reference
  implementation logged only binary target/non-target, making its data
  useless for replay. Never regress this.

## Shared interface contract

- EEG stream: `type='EEG'`, 8 ch, 250 Hz, float32, **microvolts**. Real device
  name `UnicornEEG`; the mock is `MockUnicornEEG`.
- Marker stream: `name='P300Markers'`, `type='Markers'`, 1 ch, irregular, string.
  One JSON event per sample, timestamped at `win.flip()` time:
  ```json
  {"event": "trial_start", "trial_id": 1, "target_char_idx": 15, "target_char": "ก"}
  {"event": "flash", "trial_id": 1, "flash_idx": 7, "group": [3,11,19,27,35,43], "is_target": false}
  {"event": "trial_end", "trial_id": 1}
  ```
- `collection/characters_th.json` is the single source of truth for character ↔
  index (72 entries). Load it; never hardcode a character list.
- All consumers resolve streams **by type, never by name** — that's why the mock
  works with zero changes to the real path.

## Repo map

```
start_speller.sh            # entry point: ./start_speller.sh [--demo]
collection/
  session_gui.py            # control panel: live traces, scalp-quality map, target entry, settings;
                            #   Start spawns record_session.py, then checkerboard_stimulus.py
  record_session.py         # LSL consumer -> CSV; owns EEG_CHANNELS + SANITY_PTP_MIN/MAX_UV
  checkerboard_stimulus.py  # PsychoPy matrix, cue phase, JSON markers at flip time
  flash_sequence.py         # pure checkerboard grouping logic (no PsychoPy) — unit-tested
  mock_headset.py           # synthetic EEG outlet + P300 injection on target flashes
  README_ACQUISITION.md     # machine topology + startup order
  checkerboard.py, checkerboard_experiment.py, test_sync.py   # LEGACY, pre-contract — don't use
stopping/
  bayesian_stopping.py      # log_normalize (done) + controller (3 user-owned stubs)
  calibrate.py              # Platt scaling, ECE, reliability diagram
  lm_prior.py               # ThaiNgramLM: order 3-4, add-alpha, backoff
  replay.py                 # simulate_trial, wolpaw_itr, run_sweep
  validate_offline.py       # code validation on a FABRICATED grid — not a research result
  flash_selector.py         # info-gain ordering stubs — out of current scope
classifiers/                # benchmark_all.get_pipelines() = single source of the 5 pipelines
config.py, data_loader.py   # MOABB/BNCI EDA path (volts!)
class_balance.py, signal_quality.py, erp_analysis.py, topography.py, run_eda.py
tests/                      # flash_sequence, bayesian_stopping, calibrate, lm_prior, mock_headset
```

## Running

```bash
./start_speller.sh --demo       # no hardware: mock EEG + GUI; recordings prefixed demo_
./start_speller.sh              # real headset; start the Unicorn LSL Interface FIRST
python collection/mock_headset.py --flat P3 --noisy C4 --railed O1   # exercise quality colors
python collection/checkerboard_stimulus.py --targets "กข" --n-repetitions 2 --windowed
pytest tests/ -q                # expected: 28 passed, 4 failed (the NotImplementedError stubs)
```
Conda env: `p300speller`. Recordings go to `collection/recordings/` (gitignored).

## Known gotchas — don't silently reintroduce

1. **Units split by subsystem.** The collection side (LSL, recorder, GUI,
   mock) works in **microvolts**. The MOABB path (`data_loader.py`,
   `config.PTP_REJECT_THRESHOLD_V`) works in **volts**; `load_arrays()`
   divides out `unit_factor` so it matches the Epochs path. An earlier unit
   bug inflated amplitudes ~1e6× and rejected ~100% of epochs silently.
   Never mix the two sets of thresholds.
2. **Recorder drops the first ~2.7 s of markers — UNFIXED.** `record()`
   resolves both inlets, then runs the 2 s sanity check before its first
   marker pull. pylsl doesn't buffer an inlet until it's opened, so trial 0
   loses its `trial_start` (the ground-truth label) and ~20 flashes. This was
   reproduced end-to-end with the mock, and real sessions hit it too.
   Proposed fix: call `marker_inlet.open_stream()` right after resolution.
   Awaiting user approval.
3. **Matrix orientation is unresolved.** Flat-index parity only gives a true 2D
   checkerboard when n_cols is odd. The stimulus defaults to 8×9 (odd,
   adjacency-safe); the originally stated spec was 9×8 (even, NOT safe
   without reordering characters). See `flash_sequence.py`'s docstring and
   run `validate_no_adjacent_co_flash()` before collecting real data.
4. **`characters_th.json` is `"verified": false`.** Confirm its order against
   `th_offline_cb.py` before real data collection.
5. **Montage is unverified.** `EEG_CHANNELS` = Fz, F3, C4, Cz, Pz, P3, O2, O1 is
   not g.tec's default layout (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8). Confirm it
   against the physical device. The GUI's `DISPLAY_ORDER` is separate from
   stream index order on purpose.
6. **Thai font missing.** `THSarabunNew.ttf` isn't in the repo, so the fallback
   font likely breaks tone marks. Check with `verify_thai_font_rendering()`.
7. **BNCI2014-008/009 expose no row/column identity.** Only flash-level
   metrics are valid there — no character accuracy or ITR claims.
8. **Baseline correction** (`BASELINE_WINDOW = (-0.2, 0.0)`) was added after an
   Fz-dominant sub-3 µV artifact appeared without it. Don't remove it.
9. **CV is grouped by subject, not by trial.** Don't present it as fully
   leakage-proof.
10. **~5:1 class imbalance.** Always report balanced accuracy / AUC alongside
    accuracy.
11. **FgMDM and TS+RegLDA have identical AUC — expected, not a bug.** pyriemann
    0.12's FGDA uses the same `LDA(solver="lsqr", shrinkage="auto")` on the
    same tangent-space features, so both rank epochs identically. Balanced
    accuracy differs because their decision thresholds differ.
12. **Stale references.** `character_decoder.py` is cited in docstrings but
    doesn't exist. `setup.py`'s docstring names `day2_moabb_setup.py`.
    `validate_offline.py` still prints `theta_base`.

## Results so far (flash-level, subjects 1–5, GroupKFold by subject)

From `results/all_classifiers_results.csv` (AUC / balanced accuracy):

| Pipeline | BNCI2014_008 | BNCI2014_009 |
|---|---|---|
| xDAWNcov+FgMDM | 0.703 / 0.627 | 0.824 / 0.698 |
| xDAWNcov+TS+RegLDA | 0.703 / 0.548 | 0.824 / 0.630 |
| xDAWNcov+MDM | 0.646 / 0.574 | 0.835 / 0.709 |
| xDAWNcov+MeanField | 0.640 / 0.560 | 0.835 / 0.714 |
| xDAWN+LR | 0.679 / 0.631 | 0.718 / 0.621 |

- **The best pipeline differs by dataset.** Tangent-space methods lead on 008;
  MDM/MeanField lead on 009. Report that, don't average it away.
- An earlier `benchmark_all.py` run reported FgMDM at 0.769 / 0.694 on 008,
  which this CSV doesn't reproduce. Reconcile the two run configurations
  before citing either.
- Subject 4 was the worst fold across all pipelines in earlier runs. Treat it
  as a reportable finding, not noise.

## Conventions

- No day-number prefixes in filenames.
- New pipelines go in `get_pipelines()`; classifier files stay independently
  runnable.
- Import shared constants instead of redeclaring them (`EEG_CHANNELS`,
  `SANITY_PTP_*` from `record_session`; `P300_SEARCH_WINDOW` from `config`).
- Keep pure logic separate from I/O so it's testable without hardware
  (`flash_sequence` vs `checkerboard_stimulus`, the pure functions in
  `mock_headset`). Tests on thresholds must check every channel across
  several seeds with margin — single-seed tests have already missed two
  boundary bugs here.
- Sanity-check real output against plausible ranges (scalp EEG ≈ single digits
  to a few hundred µV) before trusting or plotting it.
- Synthetic data must identify itself (`MockUnicornEEG` name, `demo_` prefix,
  GUI banner).
- Cross-check library behavior against source/docs before relying on it. That
  habit caught gotchas 1, 7 and 11.
