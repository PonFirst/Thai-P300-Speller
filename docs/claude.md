# P300 Speller Project — Context for Claude Code

## What this project actually is

PhD application portfolio work (targeting CMU ECE/BME), advised by an academic
advisor. Background: competitive programming (ICPC) / systems engineering,
building BCI + ML research depth. The advisor set an explicit project
sequence:

1. **Thai P300 BCI Speller (current priority)** — reimplementing and extending
   Pananookooln et al. (2022), the first Thai-language P300 visual speller
   (checkerboard paradigm, 8x9 matrix, 8-channel EEG). Rationale: works with
   existing 8-channel hardware, no heavy human-subjects burden, well-studied
   paradigm internationally but unexplored for Thai script — genuine novelty,
   fast path to a publishable result.
2. Motor imagery work (next, not started).
3. Meta-LaBraM (MAML on a pretrained EEG foundation model, separate project,
   further along already — crux experiment on PhysioNet data pending).
4. Real-time low-latency Transformer for closed-loop BCI (longer-term,
   sub-15ms inference target).
5. STDP-based inter-subject variability work (lower priority, needs a
   literature check before committing engineering time).

**Where we are right now**: still on step 1, specifically the "validate the
pipeline on public datasets before self-collecting Thai data" phase. Two
MOABB datasets are being used for this: `BNCI2014_008` (8ch, ALS patients,
matches the project's own hardware channel count) and `BNCI2014_009` (16ch,
healthy subjects, 3 sessions/subject — the dataset that actually supports
cross-session comparison).

## The actual end goal, don't lose sight of it

The novelty claim depends on **self-collecting Thai-character EEG data** —
pipeline validation on public English/generic-matrix datasets is
de-risking, not the contribution itself. Everything in this repo so far is
preparation for that self-collection, not the deliverable.

Two open decisions before a data-collection protocol can be written:
1. Confirm checkerboard two-color interleaving works correctly on the
   non-square 8x9 Thai matrix (don't assume square-grid logic transfers).
2. Decide the precise novelty angle: pure replication vs. Thai-script-specific
   effects (e.g. visual-complexity/adjacency-error effects on the larger
   matrix) vs. a methods contribution. **Not yet decided** — don't assume one
   on the user's behalf.

## Repo structure (current)

```
config.py                          # channel layouts, P300 search window, PTP threshold,
                                    # BASELINE_WINDOW -- single source of truth, other files
                                    # import from here rather than hardcoding constants
data_loader.py                     # load_arrays() / load_epochs() -- see "known gotchas" below
class_balance.py                   # class distribution, per-subject/session contingency tables
signal_quality.py                  # peak-to-peak artifact rejection, PSD via Welch's method
erp_analysis.py                    # grand-average ERPs, peak detection, single-trial-vs-ensemble SNR
topography.py                      # scalp topomaps at P300 peak latency
run_eda.py                         # orchestrates the four EDA modules above, CLI: --dataset --subjects

xdawn_lr.py                        # xDAWN (pyriemann) + logistic regression baseline
xdawncov_mdm.py                    # xDAWN-covariance + MDM (Riemannian, no tangent space)
xdawncov_tangentspace_mdmfamily.py # xDAWN-covariance + FgMDM + MeanField
xdawncov_ts_regLDA.py              # xDAWN-covariance + tangent space + shrinkage LDA
benchmark_all.py                   # runs all 5 classifier pipelines on IDENTICAL CV folds,
                                    # exposes get_pipelines() -- reuse this, don't redefine
                                    # pipelines elsewhere
character_decoder.py               # row/column score accumulation, character decoding, ITR --
                                    # dataset-agnostic by design, see "known gotchas" for why
```

## Known gotchas — do not silently reintroduce these

1. **Units.** MOABB's array-return path (`return_epochs=False`) multiplies by
   `dataset.unit_factor` (1e6), returning microvolts. The Epochs-return path
   stays in volts. `data_loader.py`'s `load_arrays()` divides back out by
   `dataset.unit_factor` specifically to keep both paths consistent with each
   other and with `config.py`'s volt-based `PTP_REJECT_THRESHOLD_V`. If you
   add a new data-loading path, it must also return volts, or every
   downstream `*1e6`-for-display conversion silently breaks (this exact bug
   previously inflated peak-to-peak amplitudes by ~1,000,000x and pushed
   artifact rejection to ~100% with no error thrown).
2. **Baseline correction.** `data_loader.py` constructs `P300(baseline=BASELINE_WINDOW)`
   (from `config.py`, currently `(-0.2, 0.0)`). This was added after an
   Fz-dominant, sub-3uV grand-average anomaly showed up on BNCI2014-008
   without it. Don't remove this without a reason.
3. **Row/column identity is NOT available** through `data_loader.py`'s
   standard loading path for BNCI2014-008/009. MOABB's dataset metadata only
   exposes `events={"target": 1, "non-target": 2}` — row/column stimulus
   codes live on a separate raw "Flash stim" channel that the standard
   `paradigm.get_data()` call never surfaces. This means `character_decoder.py`
   is validated only against synthetic row/column-labeled data, NOT real
   BNCI2014-008/009 data. Every classifier file gives FLASH-level metrics
   (accuracy/AUC), not character-level accuracy or ITR, on these two
   datasets. Do not claim character-level results on BNCI2014-008/009
   without first solving this (see character_decoder.py's docstring for the
   two possible paths forward). This constraint will not apply to
   self-collected Thai data, where the collection protocol will log
   row/column identity directly.
4. **Cross-validation is grouped by SUBJECT**, not by individual
   character-selection trial (MOABB's metadata doesn't expose per-character
   trial IDs for these datasets). This prevents cross-subject leakage but
   does NOT guarantee against flash-level leakage within one subject's
   session. Documented as an open limitation in `xdawn_lr.py`'s docstring —
   don't present current CV as fully leakage-proof.
5. **Class imbalance is ~5:1** (NonTarget:Target), confirmed empirically. A
   majority-class classifier gets ~83% accuracy for free. Always report
   balanced accuracy and/or AUC alongside raw accuracy — raw accuracy alone
   is misleading here and has been for every result discussed so far.

## Conventions

- **No day-number prefixes in filenames** (e.g. `xdawn_lr.py`, not
  `day4_xdawn_lr.py`) — this was an explicit correction partway through the
  project; don't reintroduce the pattern.
- Every classifier file is independently runnable via
  `python <file>.py` and is self-contained (harness code duplicated rather
  than imported across files) so any single file can be handed to someone
  else or run standalone without pulling in the rest of the repo.
- New pipelines/classifiers should go into `benchmark_all.py`'s
  `get_pipelines()` dict rather than as one-off scripts, to keep comparisons
  fair (identical CV folds across all pipelines in one run).
- Always sanity-check real output against a known-plausible range before
  trusting it (e.g. real EEG peak-to-peak amplitude is roughly single digits
  to a few hundred microvolts — a result in the millions is a red flag, not
  a number to plot and move on from). This habit directly caught the unit
  bug in gotcha #1.

## Implementation ownership

Originally the intent was for the user to implement core algorithms
(xDAWN, tangent-space mapping, etc.) from scratch for PhD-interview
readiness, with Claude/Claude Code scaffolding harnesses and reviewing
code. The user has since decided to use `pyriemann`'s existing
implementations directly for velocity, while keeping the option to
implement from scratch later. Don't assume this means rigor doesn't
matter elsewhere — cross-check library behavior against source/docs
before relying on it (this has repeatedly surfaced real, non-obvious
behavior, e.g. gotcha #1 and #3 above), and keep explanations
precise enough that the user could defend every design choice to an
advisor.

## Current best result (flash-level, cross-subject LOSO, BNCI2014-008, subjects 1-5)

xDAWNcov+FgMDM currently leads: AUC 0.769, balanced accuracy 0.694 — ahead of
xDAWN+LR (0.749 / 0.688), xDAWNcov+MDM (0.719 / 0.641), and xDAWNcov+MeanField
(0.717 / 0.629). `xdawncov_ts_regLDA.py` has not yet been run on real data —
don't assume FgMDM is final without it.

One consistent finding across every pipeline tried so far: subject 4 is the
worst-performing fold regardless of classifier — worth treating as a real,
reportable finding (check subject 4's Day-3-era EDA output: ERP amplitude,
rejection rate) rather than dismissing as noise.