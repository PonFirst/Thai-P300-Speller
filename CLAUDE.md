# Thai P300 Speller — Context for Claude Code

## What this is

PhD application portfolio work (targeting CMU ECE/BME): a Thai-language P300 visual speller extending Pananookooln et al. 2022 (AIT Brain Lab). Checkerboard paradigm, 72 Thai characters, 8-channel EEG at 250 Hz (g.tec Unicorn Hybrid Black).

- **The classifier** (xDAWN + RegLDA, `classifiers/`) is validated and not the contribution.
- **The contribution** is Bayesian dynamic stopping (`stopping/`): a fixed confidence threshold plus a Thai character-level LM prior.
  - The LM is a character-level Transformer, now the intended production prior. It was trained in `stopping/gpt_dev.ipynb` (Karpathy-style GPT, full `pythainlp/thai-wiki-dataset-v3` corpus, run on Colab) and is wired in as the `transformer` `language_model` type: `pipeline/manifest.py` schema, `_build_transformer` in `pipeline/language_model.py`, model code in `stopping/transformer_lm.py`, selected at fit time with `--lm-transformer-checkpoint`.
  - The n-gram (`stopping/lm_prior.py`, type `ngram`) is kept as a registered alternative and a comparison baseline; `uniform` is the no-LM control.
  - The model trains on natural, unrestricted Thai text and is **marginalized** to the 72 matrix characters only when a prior is produced (`get_prior`), same convention as the n-gram.
- **Boundary-aware thresholding is descoped.** The paper's novelty angle is **not decided**; don't assume one.
- **Predictive whole-word matrix cells are rejected — do not propose this again.** The idea: replace some fixed single-character cells with LM-predicted word completions, to commit multiple characters in one trial. Explored and rejected on its merits, not just "too much work":
  - `flash_sequence.build_flash_sequence()` hard-requires an even `n_chars` (splits into two equal checkerboard colors) and `n_cols` odd (adjacency-safety); the matrix can't just shrink by an arbitrary amount.
  - A word-completion "probability" and a next-character probability aren't the same measured quantity and can't coherently share one normalized `get_prior()` distribution without a real reformulation of what the LM prior even means for those slots.
  - It needs a dynamically-rebuilt-per-trial matrix; everything today (the manifest's sha256-verified `characters` file, `board_schedule.py`'s pre-built whole-session flash schedule) assumes static, fixed-in-advance matrix content.
  - Whole-word generation (not just next-character prediction) is a materially harder, separate ML task from the character-level Transformer already built.
  - The character-level LM prior already delivers the same underlying mechanism (biasing the starting posterior toward the likely next letter) with none of this risk — this idea's marginal benefit on top of that, even if it worked, was never established.
  - If it resurfaces, it's future-work for the paper's discussion section, not something to build here.

## Hard constraints — do not relitigate

- **Hardware path.**
  - **The study runs on the Mac (owner, 2026-10-01).** Don't propose Windows for the study's own sessions or as a fallback. The old Windows files for the Python stack (`setup_windows.sh`, `WINDOWS_SETUP.md`, `SETUP.txt`) are legacy.
  - **This is the `windows` branch (owner, 2026-10-06):** the BCI2000 path ported to Windows so other people can run the experiment on a Windows laptop. `UnicornSerialSource` reads a COM port there (`ByteSourceWin32.cpp`), and `bci2000/run_session.ps1` is the launcher. Users get a prebuilt zip (BCI2000 + modules + private Python + `models/thai_gpt.lmbin`) from `.github/workflows/windows-package.yml`; they follow `START_HERE.txt` and double-click `ThaiSpeller.cmd` (menu: `bci2000/windows/menu.ps1`). No PyTorch on Windows: the Transformer runs from the `.lmbin` in NumPy (`stopping/thai_gpt_numpy.py`). Maintainer notes: `bci2000/WINDOWS.md`. **Written on the Mac; nothing has run on Windows until the workflow's first rehearsal.** Keep it in step with `bci2000-port`: merge that branch in, don't fork the shared code.
  - **Moving to BCI2000 (owner's professor, 2026-10-01).** The professor rejected the custom acquisition and stimulus stack; the project is being rebuilt on BCI2000, which is a source build at `~/BCI2000/trunk`. Custom logic is ported to C++ BCI2000 modules in `bci2000/` (owner authorized the C++ conversion). Each port is tested against its Python original as the reference. Until the port is done, the existing Python stack still works, but it is not the study's path.
  - **Headset → BCI2000 on the Mac: `bci2000/UnicornSerialSource`.** It stands in for BCI2000's `gUnicornADC`, which needs g.tec's Windows-only `Unicorn.dll`. It reads `/dev/cu.UN-*` using g.tec's documented protocol and fills lost packets so BCI2000's sample-count clock stays aligned (`Validation = 0` on filled samples). Tested against the protocol-accurate fake through real BCI2000 (`tests/test_bci2000_unicorn_source.py`); **not yet run against the real headset.** BCI2000's stock `LSLSource` is built too, but it discards LSL timestamps and does no gap filling, so don't use it with the relay for recordings.
  - BrainFlow's Unicorn class is compiled out on macOS, so nothing here uses BrainFlow or a g.tec SDK.
  - **The Python relay** (`collection/unicorn_relay.py` → LSL `UnicornEEG`) serves the pre-BCI2000 stack. With the real headset on 2026-09-30 it passed H1 and H4 (2 min: 0 non-monotonic steps, 250.05 Hz, 3 gaps totalling 114 samples in one 1.5 s burst). H12–H14 are still open.
  - Never suggest `pip install brainflow`.
- **The core stopping math is owner-implemented and now done.** `BayesianStoppingController.reset()`, `.llr()`, `.update()` were filled in by hand (blend-with-uniform-floor prior, Platt-scaled LLR with the pi correction term, group-only log-domain update + fixed-threshold stop) and are verified against a hand-worked toy example in `tests/test_bayesian_stopping.py`.
  - Never implement, rewrite, or "fix" them without an explicit ask — that protection doesn't end just because they're filled in now.
  - Everything around them is fair game.
- **The inference service is a strict checkpoint loader, never a fallback trainer.** It reads the classifier path from `session_manifest.json` and refuses to start (exit 2) unless the checkpoint's channel order, sample rate, and epoch window match the live stream exactly. Never relax this "to get something running" — tests use a scripted/mocked classifier for exactly this reason, and real inference stays blocked until a real checkpoint exists from `pipeline/fit_from_recording.py`. That fit is per-subject calibration, not "retraining the algorithm" — no algorithm redesign is implied, and fitted weights are never expected to transfer to new hardware without a new fit.
- **No dependency on g.tec's Unicorn Speller, ever, for any reason** — not a fallback, not a comparison baseline, not a customization target. The stimulus, classifier, calibration, LM prior, and stopping logic are fully custom and stay fully custom. The only g.tec software this project touches is the Unicorn LSL Interface (Windows only), a raw data relay with no P300 logic of its own; on a Mac it touches none, only g.tec's published Bluetooth protocol document. A task that seems to need Unicorn Speller needs to be solved in this project's own code instead.
- **The montage is `Fz, C3, Cz, C4, Pz, PO7, Oz, PO8` — the Unicorn Hybrid Black's factory-default cap, now verified, not the reference paper's layout.** Confirmed against g.tec's own Unicorn Suite Hybrid Black user manual (github.com/unicorn-bi/Unicorn-Suite-Hybrid-Black-User-Manual), cross-checked against a MathWorks third-party support-package listing and multiple published papers using this exact headset. The earlier assumption (`Fz, F3, C4, Cz, Pz, P3, O2, O1`) was inherited from Pananookooln et al.'s own EEG system and cap and was never actually checked against this project's hardware — four of the eight positions are genuinely different scalp sites, not relabelling: C3 vs. the old F3, PO7 vs. P3, Oz vs. O2, PO8 vs. O1. `pipeline/constants.EEG_CHANNELS` is the single definition; every other file imports it. Any recording or checkpoint made before this correction used the wrong channel identities on four of eight channels and is not safe to fit or evaluate against — treat any pre-correction `collection/recordings/*.csv` or `models/*` as invalid, not as a shortcut. Don't drift back toward the reference paper's montage.
- **The Transformer LM is trained on Colab, never on this Mac.** The owner's instruction, verbatim: "DO NOT run it in my computer, it will EXPLODE." `stopping/gpt_dev.ipynb` downloads the ~430 MB corpus and trains for ~25 min on a T4; do not run it, or a scaled-down version of it, locally. Local work with it is limited to loading a finished checkpoint and small forward passes (`ThaiTransformerLM.get_prior` takes single-digit ms on CPU).
- **A Transformer checkpoint is only valid for the matrix it was trained against.** Its vocab force-includes every character of `characters_th.json` as it was at training time. `ThaiTransformerLM` raises `ValueError` at construction if a matrix character is missing from the checkpoint's vocab, and `get_prior` indexes directly instead of falling back to `<unk>`. Never restore a fallback: it yields a distribution that sums to 1 but is silently wrong. If the matrix changes again (it changed once already: digits 6/7 → ั ็), retrain on Colab with the new file.
- **The inference service never decodes with the cued target.** It reads `target_char_idx` from the `trial_start` marker only to score commits in the experiment log (`correct`, `target_prior_rank`). Keep that read log-only.
- **Every recording logs full per-flash metadata.** Never regress this.
- **Don't modify the stimulus or the classifier without an explicit ask.**
  - `collection/checkerboard_stimulus.py` was out of scope in the last round.
  - `classifiers/` is validated: fit it, don't change it.

## Environment

- A pipenv virtualenv (`storefront`) sits earlier on `PATH` than conda, so **`python` is the wrong interpreter even after `conda activate p300speller`**. Always call the env's interpreter by path: `PY="$(conda info --base)/envs/p300speller/bin/python"`.
- `start_speller.sh` does this for you, and the GUI launches its child processes with `sys.executable`.
- GUI tests need `QT_QPA_PLATFORM=offscreen`.
- `stopping/` and `pipeline/` are packages: run their modules with `$PY -m stopping.<name>` from the repo root. `$PY stopping/<name>.py` fails with `No module named 'stopping'`.
- `torch` and `jupyter` are in `requirements.txt` for the Transformer LM. Checkpoints are gitignored: `models/thai_gpt_checkpoint.pt` is the full-corpus model (5.60M params, block 64, 1,661-token vocab) and the one to use; `stopping/data/thai_gpt_checkpoint.pt` is the superseded first pass (0.88M params, block 128), kept only for comparison.

## Contracts

`docs/INTERFACES.md` is the reference for how the pieces talk to each other. Code in `pipeline/` defines the contracts; import it, never redefine it.

| What | Where |
|---|---|
| Stream names/types, channel order, 250 Hz, µV thresholds, the dejitter switch | `pipeline/constants.py` |
| `P300Inference` events (service → GUI), strictly validated on both ends | `pipeline/events.py` |
| `session_manifest.json`: which model files a run uses, LM included (`uniform` / `ngram` + corpus / `transformer` + checkpoint; sha256-verified, fingerprinted) | `pipeline/manifest.py` |
| Experiment logs: sidecar file names, `<csv>.session.json` (calibration sentence, timeline), `models/<id>_fit_report.json`, inference-log records | `pipeline/experiment_log.py` (schema table in `docs/INTERFACES.md`) |
| Quality levels + quality gate (shared by GUI map and service) | `pipeline/quality.py` |
| Epoch cutting (shared by fit script and service: train/serve parity) | `pipeline/epoching.py` |
| Marker JSON (`trial_start` / `flash` / `trial_end`) | `collection/checkerboard_stimulus.py` |
| Recorder CSV + gap rows + `<csv>.log.jsonl` + `<csv>.integrity.json` | `collection/record_session.py` docstring |

- **Resolution:** all consumers find streams **by type**, and every tool takes `--eeg-type` / `--marker-type` / `--inference-type`.
- **`P300Inference` stream type:** it is deliberately not `Markers`.

## Repo map

```
bci2000/                         this project's BCI2000 modules (C++), symlinked to ~/BCI2000/trunk/src/custom;
                                 README.md has build/run steps
  UnicornSerialSource/           headset source module: UnicornProtocol (port of unicorn_protocol.py), ByteSource
                                 (serial/TCP; ByteSource.cpp macOS, ByteSourceWin32.cpp Windows COM port + Winsock),
                                 UnicornSerialADC (BufferedADC, gUnicornADC-compatible)
  CheckerboardSpeller/           application module: CheckerboardSequence.h (port of flash_sequence.py),
                                 CheckerboardSpellerTask (StimulusTask; per-flash codes, FlashChar1-6 states),
                                 BayesianStopping.h (port of bayesian_stopping.py), ThaiGPT.h (Transformer prior)
  CheckerboardSignalProcessing/  SP module: RiemannP300Classifier (epochs + sosfiltfilt + xDAWNcov/TS/LDA per flash,
                                 RiemannP300Math.h = Eigen port of pipeline/bci2000_export.reference_score)
  run_session.sh                 one BCI2000 run: check | calibrate | online [--condition transformer|uniform|fixed];
                                 --fake --auto rehearse without the headset
  run_session.ps1, session_helper.py   the Windows launcher (headset port found automatically) and its helper
  windows/                       menu.ps1 (the session menu ThaiSpeller.cmd starts), sentences.json,
                                 requirements.txt (the package's Python), NOTICE.txt; WINDOWS.md = maintainer notes
  parms/ThaiCheckerboard.prm     GENERATED by tools/make_bci2000_parms.py from characters_th.json
start_speller.sh                 ./start_speller.sh [--demo] [--manifest PATH] -> Qt panel
setup_windows.sh                 LEGACY (Mac only now): Windows env setup
WINDOWS_SETUP.md                 LEGACY (Mac only now): Windows walkthrough
collection/
  session_gui_qt.py              PySide6 panel: traces, scalp map, target entry, settings, composed text,
                                 confidence gauge; launches service -> recorder -> stimulus
  gui_logic.py                   all panel logic without Qt (tested): settings, paths, connection state,
                                 inference display model, SessionOrchestrator, SessionLogger (writes <csv>.session.json)
  board_qt.py, board_schedule.py in-window stimulus board (the panel's default) and its timeline;
                                 same markers and trial structure as checkerboard_stimulus.py
  session_gui.py                 LEGACY Tkinter panel, kept as a fallback
  record_session.py              hardened LSL recorder (backoff, gap rows, integrity checks, JSON logs)
  checkerboard_stimulus.py       PsychoPy matrix, cue phase, JSON markers at flip time
  flash_sequence.py              pure checkerboard grouping logic
  mock_headset.py                synthetic EEG outlet: fault flags, P300 on target flashes, fixed sample grid
  unicorn_relay.py               Mac path: Unicorn Bluetooth serial (or --tcp) -> LSL UnicornEEG + UnicornAux;
                                 counter-based timestamps, counter-gap report (--duration/--report)
  unicorn_protocol.py            pure logic for it: payload decode/encode, stream framer, counter tracker, CounterClock
  README_ACQUISITION.md          single-machine setup (primary) + startup order, recorder output and exit codes
  checkerboard.py, checkerboard_experiment.py, test_sync.py   LEGACY, pre-contract; don't use
pipeline/
  constants.py quality.py events.py manifest.py epoching.py experiment_log.py   shared contracts
  inference_service.py           live inference: LSL in, P300Inference out (InferenceEngine is pure); writes the
                                 experiment log (LM prior at each character's start, per-flash updates, commits)
  fit_from_recording.py          calibration CSV -> classifier checkpoint + Platt calibration + manifest
                                 + fit report; --lm-corpus (ngram) | --lm-transformer-checkpoint (transformer)
  classifier_io.py               checkpoint save/load, stream compatibility, channel-label policy
  bci2000_dat.py                 BCI2000 .dat reader (format 1.1), checked against BCI2000Tools.FileReader
  bci2000_recording.py           CheckerboardSpeller .dat -> flashes (onset/label/selection), lost samples -> NaN
  bci2000_export.py              fitted checkpoint -> BCI2000 .prm for CheckerboardSignalProcessing; reference_score()
  bci2000_lm_export.py           Transformer checkpoint -> .lmbin weights for the speller's ThaiGPT.h
  bci2000_session.py             CheckerboardSpeller .dat (+ .applog priors, matched by SpellerRunID) ->
                                 inference-log records, so summarize_experiment reads BCI2000 sessions
  language_model.py              LM factory (uniform | ngram | transformer), prior validated at the boundary
stopping/
  bayesian_stopping.py           log_normalize + controller: reset/llr/update, owner-implemented, done
  calibrate.py lm_prior.py replay.py   Platt/ECE, ThaiNgramLM, replay + Wolpaw ITR
  transformer_lm.py              ThaiGPT architecture + ThaiTransformerLM (checkpoint -> get_prior); single copy
  gpt_dev.ipynb                  trains the Transformer LM: RUN ON COLAB ONLY (see hard constraints)
  compare_transformers.py, model_comparison.py   old vs new Transformer vs n-gram: accuracy, latency (CPU)
  summarize_experiment.py        logs -> accuracy + Wilson CI, Wolpaw ITR, LM effect, per-character table
  evaluate_recording.py          real-recording replay: accuracy/mean-flashes/ITR swept over theta/t_min/...
  validate_offline.py            code validation on a FABRICATED grid, not a research result
  flash_selector.py              info-gain ordering stubs, out of scope
tools/
  fake_inference_publisher.py    scripted P300Inference stream for GUI QA
  e2e_demo_smoke.sh              no-hardware end to end: record -> fit -> service ready (opens a window)
  fake_unicorn_bt.py             protocol-accurate fake headset over TCP (drops, bursts, garbage); its data is
                                 marked synthetic and the relay then names the streams Mock*
classifiers/                     benchmark_all.get_pipelines() = single source of the 5 pipelines
config.py, data_loader.py        MOABB/BNCI EDA path (volts!)
tests/                           see TESTING.md; test_fault_injection.py = cross-component scenarios
docs/INTERFACES.md, TESTING.md
```

## Running

```bash
./start_speller.sh --demo                       # no hardware: mock EEG + panel; recordings prefixed demo_
./start_speller.sh                              # real headset; start the Unicorn LSL Interface (Windows) or the relay (Mac) FIRST
$PY collection/unicorn_relay.py                 # Mac: headset paired in System Settings -> /dev/cu.UN-*; --list-ports to check
$PY collection/unicorn_relay.py --duration 1800 --report logs/gap_test.json      # the H12 sample-counter gap test
$PY tools/fake_unicorn_bt.py --port 9000        # no headset: then  $PY collection/unicorn_relay.py --tcp 127.0.0.1:9000
$PY collection/mock_headset.py --flat PO7 --noisy C4 --railed Oz      # exercise quality colours
$PY -m pipeline.fit_from_recording --recording collection/recordings/<file>.csv --id subject_001 \
    --lm-transformer-checkpoint models/thai_gpt_checkpoint.pt     # omit the flag for the uniform LM; --lm-corpus for ngram
$PY -m pipeline.fit_from_recording --recording <calibration R01>.dat [<R02>.dat ...] --id subject_001 \
    --lm-transformer-checkpoint models/thai_gpt_checkpoint.pt   # also writes models/<id>_bci2000.prm + <id>_lm.lmbin
bci2000/demo_speller.sh [--slow] [TEXT]         # BCI2000 speller with the fake headset (opens a window)
bci2000/run_session.sh check --subject S01      # then calibrate / online; see bci2000/README.md
$PY tools/h13_montage_check.py <check run>.dat  # H13 montage + H3 units (CSV works too)
$PY -m pipeline.inference_service --manifest session_manifest.json    # normally launched by the panel
$PY -m stopping.summarize_experiment --recording collection/recordings/<test_session>.csv   # accuracy, ITR, LM effect (.dat too)
$PY -m stopping.model_comparison                # old vs new Transformer vs n-gram (fits the n-gram: ~16 s)
$PY tools/fake_inference_publisher.py --include-malformed             # drive the composed-text panel
QT_QPA_PLATFORM=offscreen $PY -m pytest tests/ -q
```

- **Expected tests:** all pass. Three tests self-skip with "BayesianStoppingController.reset() is implemented now" (`test_fault_injection.py`, `test_inference_service.py` x2) — they exist to check stub-only behavior and correctly stand down now that it's filled in.
- **Output locations:** recordings go to `collection/recordings/` and models to `models/` (both gitignored). Demo artifacts are always `demo_`-prefixed.
- **Experiment logs** sit next to each recording: `<csv>.session.json` (the calibration sentence typed, settings, timeline), `<csv>.log.jsonl` + `<csv>.integrity.json` (recorder), `<csv>.inference.log.jsonl` (LM prior per character, per-flash updates, commits). The fit report is `models/<id>_fit_report.json`. Nothing in `collection/recordings/` is created by tests.

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
9. **The montage is now verified.** `EEG_CHANNELS` = Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, the Unicorn's factory-default layout (see the hard-constraints entry above for sources). It replaced an unverified assumption inherited from the reference paper's own cap; don't reintroduce those old names as this project's channel identity.
   - The real device's own reported channel labels (as opposed to the assumed layout) are still unconfirmed against a live stream — H2 in TESTING.md is still open.
   - The service warns on generic labels and refuses when the electrode names differ.
   - See TESTING.md H1/H2.
10. **Thai fonts.**
    - The GUI uses Sarabun, falling back to Thonburi; its tone-mark stacking is verified by an automated test.
    - The PsychoPy stimulus now bundles the genuine THSarabunNew.ttf (`collection/fonts/`, SIPA's official public font, GPL-2.0+Font-exception) and `_resolve_font`'s default points at it. Visually verified: every grid cell renders one isolated character correctly (bare combining marks show the standard dotted-circle placeholder, which is correct, not broken). `TextStim` does NOT shape a multi-character combining sequence correctly (confirmed: `"ก่อ"` as one string detaches the tone mark) -- irrelevant today since no cell ever concatenates characters, but means composed Thai text must go through `TextBox2` instead if it's ever needed here, not `TextStim`.
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
19. **`get_prior()` runs once per trial, not per flash.** `InferenceEngine._start_trial` calls it at `trial_start`; the controller's `update()` then runs once per flash and touches only that flash's 6-character group (one row or column of a color's 6×6 virtual matrix; `pi` = 6/72), then renormalizes all 72. The latency budget is therefore the stimulus's `cue_duration_ms` (2000 ms), not the 124 ms flash interval.
20. **The controller's `lam` blend caps how much the LM can move the start.** `p0 = lam·prior + (1−lam)/72`, so with the default `lam = 0.7` a 99% LM prior starts the posterior near 0.70. The experiment log records `lm_prior` and `initial_posterior` side by side so this stays visible.
21. **The Transformer's `<unk>` token is U+E000 (private use).** Terminals and editors often draw it as an empty string (`UNK = ""` in a diff), but the stored value is correct. Check with `repr()` or `ord()` before "fixing" it.
22. **Transformer LM latency and accuracy figures are Mac-only and small-sample.** Measured on an M3 Max, CPU forced: 6.7 ms per `get_prior` (≈300× under the 2000 ms budget). The Windows deployment machine is unmeasured. Accuracy comes from 16 hand-picked common-word completions, which is suggestive, not a publishable evaluation.

23. **With the relay, the channel labels are ours.** The headset sends none, so `unicorn_relay.py` writes `EEG_CHANNELS` onto channels 1–8, and the service's electrode-label check can no longer catch a wrong cap. Verify physically (TESTING.md H13); H2's "real device labels" question is moot for this path.
24. **The relay's timestamps come from the device counter, not arrival.** Bluetooth bursts would otherwise smear ERPs (simulated: ~0.1 ms stamp error vs ~5 ms on arrival). A lost packet is a stamp jump that the recorder already detects, and `samples.missing` in the relay report is the true loss. The constant link delay is unmeasurable in software (H7, H14). Cost of the relay itself: ~7 µs per packet.
25. **g.tec's protocol PDF has a typo:** its counter formula prints `Counter[1]` three times; the C example beside it (and `unicorn_protocol.py`) use bytes 0–3, little-endian. EEG is big-endian 24-bit; accelerometer, gyroscope and counter are little-endian.
26. **Mac display timing (from the owner's research notes, measured on an M3 Max in a separate session; not re-verified in this repo).** Reported: steady flashing locks at 8.33 ms frames, so the PsychoPy stimulus's 62 ms waits round up to 66.7 ms flashes and a 133.3 ms SOA at both 60 and 120 Hz (the paradigm never ran at 62/124 ms on any machine); ProMotion's idle slowdown leaves the first flash after a static screen a few ms uncertain; the in-window Qt board keeps 124 ms, so **the two boards give different SOAs: calibrate and run online on the same board.** For sessions: fixed 60 Hz (System Settings → Displays; unmeasured), charger plugged in, "Separate window" board. `evaluate_recording` is unaffected because it measures time per flash from the recording's markers. Flip-to-light lag is still unknown (H7).

27. **BCI2000 speller: one stimulus code per flash needs `SingleEpochMode = 1`.** The checkerboard reshuffles its groups every repetition, so `CheckerboardSpeller` numbers flashes 1..24×repetitions within each selection. Two things follow:
    - **The bug this avoids:** BCI2000's `P3TemporalFilter` with `EpochsToAverage = 1` but `SingleEpochMode = 0` keeps counting each code's epochs across selections. It reports only when the count equals 1 and resets only at `NumberOfSequences`, so every other selection silently got zero classifier results (found in an end-to-end run). The parameter file sets it, and the module's preflight refuses to start otherwise.
    - **Analysis:** the per-flash group lives in the `FlashChar1`–`FlashChar6` states (1-based character indices), not in the stimulus code.

28. **BCI2000 scripts: pause between `Wait for Resting` and `Start`.** Parameter messages sent during configuration can still be queued when a script starts the run straight away. If one reaches a module that is already running, the module raises `Unexpected Param message` and the whole system shuts down; the script then reports `Wait aborted`.
    - **Measured:** 2 of 6 demo runs failed this way; with `Sleep 2` before `Start`, 8 of 8 were clean. The demo script and the BCI2000 tests now include the pause. A person clicking Start in the Operator is slow enough not to hit it.
    - **Also:** modules must not write parameters in `Initialize` (`gUnicornADC` writes `ID_Amp` there; `UnicornSerialSource` doesn't).
    - **Diagnosis:** pass `--SystemLogFile=<path>` on the `Startup system` line to get the module errors.

29. **BCI2000 classifier: train/serve parity by construction, not by retraining.** `CheckerboardSignalProcessing` doesn't use BCI2000's causal online filters. It cuts each flash's padded window and runs scipy's `sosfiltfilt` on it, exactly as `pipeline/epoching.py` does offline. Three consequences:
    - **Results arrive late:** 1.3 s after each flash (0.8 s epoch plus 0.5 s pad).
    - **`EpochLength` must cover that (1400 ms).** BCI2000's StimulusTask decides `EpochLength` after the last flash, and results arriving later were counted toward the next character: found in an end-to-end run, selection 1 decided on 39 of 48 results. The filter's preflight refuses a shorter value.
    - **Onset alignment is identical online and in the `.dat` (measured).** The SP module sees `StimulusCode` at the same sample index the recording stores.
30. **The BCI2000 path rejects exactly what the Python paths reject, and scores the rest identically.**
    - **The quality gate:** `pipeline/quality.py`'s rule, ported in C++ with its thresholds exported into the model `.prm` (`RiemannQualityGate`).
    - **Then the rank check:** `pipeline/epoching.is_rank_deficient`. Linearly dependent channels, such as bridged electrodes the gate can't see, make the covariance singular, and sklearn then raises or returns arbitrary scores.
    - **Lost samples:** `Validation = 0`, plus the first sample after any device `Counter` break.
    - **Applied by the service, `evaluate_recording`, the fit and the filter:** all four use this order. The service and `evaluate_recording` gained the rank check; before, a flat channel under the `warn` policy crashed the service.
    - **Selection:** the speller selects the character with the highest **mean** score over its scored flashes. BCI2000's default **sum** counted a rejected flash as 0, which is not neutral for an LDA score and biased selections toward characters whose flashes were dropped.
    - **What counts as real:** a `.dat` is real only if it proves it: source `UnicornSerialADC`, no `TcpAddress`, and gyroscope states without the fake's `0x5A5A`. Anything else is demo data.
31. **BCI2000 Bayesian stopping is the Python controller, ported.**
    - **The C++:** `BayesianStopping.h` is ported line by line from the owner-implemented `reset`/`llr`/`update` (C++ port authorized). The Python stays the reference.
    - **Parity is tested:** in random sequences (`tests/test_bci2000_stopping.py`) and in closed-loop BCI2000 sessions replayed in Python (`test_bci2000_stopping_e2e.py`).
    - **The one tolerated difference:** characters tied in exact arithmetic, i.e. same flashes under a uniform prior. numpy's vectorised `log` rounds them an ulp apart and the C++ keeps them equal, so the reported argmax among them can differ.
    - **Defaults:** the speller's `StoppingRule` is 1 and `LanguageModel` is 0 (uniform). The fit's model `.prm` sets the Transformer (`<id>_lm.lmbin`) and the manifest's stopping values.
    - **With BCI2000's stock P3 classifier:** use `StoppingRule 0`. The speller refuses Bayesian stopping there, because it has no Platt calibration.
32. **BCI2000 parameter files: escape `{` and `}` in string values.** BCI2000 reads braces as list delimiters even inside a string, and an unescaped `{` in the exported JSON provenance swallowed the following parameter lines, leaving `RiemannChannels` empty. Both encoders (`bci2000_export._bci_encode`, `make_bci2000_parms.encode`) write them as `%7B` / `%7D`.

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

## LM prior comparison (`stopping/model_comparison.py`; 16 hand-picked Thai completions, CPU on an M3 Max)

| LM | top-1 | top-3 | mean surprisal | mean `get_prior` | one-time cost |
|---|---|---|---|---|---|
| Transformer, full corpus (5.60M params) | 81.3% | 100% | 0.63 bits | 6.7 ms | 46 ms load |
| N-gram, order 4 | 75.0% | 100% | 0.96 bits | 0.008 ms | 15.8 s fit |
| Transformer, first pass (0.88M params) | 68.8% | 93.8% | 1.18 bits | 2.0 ms | 10 ms load |

- With 16 items the ranking is suggestive only. To compare LMs on the actual task, replay one recording under each (`stopping.evaluate_recording`); each model's own training loss is not comparable (different corpus, vocab, split).
- Training-run facts, for the paper: full corpus, 20,000 steps, batch 64, block 64, 6 layers, 256-dim, 8 heads, ~25 min on a Colab T4, final train 1.53 / val 1.16 (val stayed below train throughout, so no sign of overfitting). Batches are sampled at random with replacement: 20,000 × 64 × 64 ≈ 82M positions. The full-corpus character count was not recorded, so how many passes that is over the data is unknown; the notebook's `train: … tokens` line on Colab has it.

## Conventions

- **Filenames:** no day-number prefixes.
- **Pipelines:** new ones go in `get_pipelines()`.
- **Shared constants:** import them; don't redeclare. That includes the experiment-log file suffixes and schemas (`pipeline/experiment_log.py`) and the Transformer architecture (`stopping/transformer_lm.py`).
- **Pure logic apart from I/O:** keep it separate, and test it without hardware.
  - Examples: `flash_sequence`, `gui_logic`, `InferenceEngine`, `mock_headset`'s pure functions.
- **Threshold tests:** check every channel across several seeds, with margin. Single-seed tests have missed two boundary bugs here.
- **Real-LSL tests:** use randomized stream types, never the production ones.
- **Tests never write into the repo.**
- **Synthetic data must identify itself:** `MockUnicornEEG`, the `demo_` prefix, the GUI banner.
- **Sanity checks:** check real output against plausible ranges before trusting it.
- **Library behaviour:** check it against source or measurement before relying on it. That habit found gotchas 1, 3, 5 and 15.
