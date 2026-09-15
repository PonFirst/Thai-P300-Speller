# How the Thai P300 Speller works, end to end

A plain-language walkthrough of the whole system: the machines involved, what
each process does, and the full path from "a flash happens" to "a character
appears on screen." For the authoritative contracts (exact schemas, field
names, timeouts), see `docs/INTERFACES.md` — this file explains the *why* and
the *flow*; that one is the source of truth for the *exact shape*.

---

## 1. The big picture

A P300 speller lets someone "type" by looking at the character they want
while a grid of characters flashes in groups. Each time the target
character's group flashes, their brain produces a P300 wave — a positive
voltage deflection roughly 250-550 ms after the flash — that a non-target
flash does not produce. Averaged/accumulated over enough flashes, the
system can tell which character the person is attending to and "type" it,
with no muscle movement required.

This project's speller uses **72 Thai characters in an 8×9 grid**, a
**checkerboard flashing pattern** (not simple row/column flashing — see §3),
an **xDAWN + tangent-space + shrinkage-LDA classifier** to score each flash,
and a **Bayesian dynamic-stopping controller** that decides, flash by flash,
whether it's confident enough to commit a character yet, instead of always
waiting for a fixed number of repetitions.

### Why the headset driver dictates the deployment machine

The Unicorn Hybrid Black headset's driver only runs on Windows or Linux —
BrainFlow's Unicorn support is compiled out on macOS entirely, and g.tec's
own "Unicorn LSL Interface" app is Windows-only. So real sessions run on
**one Windows machine the project owner fully controls**: it runs the
headset driver, the Unicorn LSL Interface, and every process in this
repo — the recorder, the live inference service, and the GUI (which in turn
launches the stimulus) — all together, talking over LSL on localhost. (This
repo is developed on a separate Apple Silicon Mac, which never touches the
device or runs a real session.)

```
One Windows machine
+---------------------------------------------------------------+
| Unicorn headset (USB) -> Unicorn LSL Interface                |
|                            -> publishes "UnicornEEG" (8ch,250Hz)|
|                                        |                        |
|                                        v      LSL (localhost)   |
| checkerboard_stimulus.py  ------> record_session.py            |
|  -> publishes "P300Markers"        (resolves BOTH outlets,      |
|                                      merges, writes one CSV)    |
|                                                                 |
| collection/session_gui_qt.py drives all of the above           |
| pipeline/inference_service.py (optional, live inference)       |
+---------------------------------------------------------------+
```

Every process finds streams **by type**, never by hostname, so this is a
deployment simplification rather than an architectural one — an older setup
that split the headset driver onto a separate VM or Linux box, talking over
the network instead of localhost, still works unchanged if ever needed
again (`collection/README_ACQUISITION.md` documents it as a variant). Every
piece of code in this repo is a pure LSL *consumer* either way — it never
talks to the headset directly, on any machine.

There are three LSL streams in total, and every process finds them **by
type**, never by name, so a mock headset or a differently-named outlet is
picked up automatically:

| Stream | type | Carries |
|---|---|---|
| `UnicornEEG` | `EEG` | 8 channels (Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 — the Unicorn's factory-default cap), 250 Hz, microvolts |
| `P300Markers` | `Markers` | One JSON event per stimulus event: `trial_start`, `flash`, `trial_end` |
| `P300Inference` | `P300Inference` | One JSON event per inference decision: posterior updates, committed characters, rejections, status |

---

## 2. The three phases of using the system

1. **Calibration (training data collection).** The subject looks at a
   sequence of known target characters while the matrix flashes. Because the
   target is known in advance, every flash can be labeled target/non-target.
   This recording is later used to *fit* the classifier — the subject hasn't
   "spelled" anything freely yet, they're just providing labeled examples.
2. **Fitting.** Offline, on the calibration recording: cut out the EEG epoch
   after each flash, train the classifier to distinguish target vs.
   non-target epochs, and calibrate its raw scores into real probabilities.
   This produces a *session manifest* — a pointer to a specific, verified set
   of model files.
3. **Live spelling (inference).** With a fitted manifest loaded, the subject
   now looks at whatever character they actually want to type. The live
   inference service scores each flash in real time and the Bayesian
   stopping controller decides, after enough evidence has accumulated,
   which character to commit — without knowing the target in advance.

The GUI panel (`collection/session_gui_qt.py`) drives all three: the same
"type target characters, press Start" flow is used for calibration (targets
known) and the same flow works for live spelling once the operator ticks
"Live inference" and points it at a manifest.

---

## 3. Data acquisition: from electrode to CSV row

### 3.1 The stimulus decides what flashes, and when

`collection/checkerboard_stimulus.py` (or the in-window equivalent,
`collection/board_qt.py`) opens a full-screen 8×9 grid of the 72 Thai
characters. For each character the subject is meant to attend to (the
"trial"):

1. **Cue phase (2000 ms default):** the target character is highlighted so
   the subject knows what to focus on.
2. **Flash phase:** groups of characters flash white/black in a
   **checkerboard pattern**, not plain rows/columns. Characters are split
   into two colors by flat list-index parity, then each color's characters
   are reshaped into a virtual 6×6 grid and flashed by virtual row, then by
   virtual column. Every character appears in exactly 2 flash groups per
   repetition (one row-group, one column-group of its own color), and this
   repeats for `n_repetitions` (5 by default) — so a full trial is
   `2 × 5 = 10` flash groups' worth of *sequences*, each sequence containing
   several individual flashes (see `collection/flash_sequence.py`). Flashes
   are 62 ms on, 62 ms off — about 124 ms apart.
   - **Why checkerboard, not row/column:** it avoids two adjacent characters
     always flashing together, which otherwise degrades classification
     ("adjacency error"). The pure list-index-parity trick only produces a
     true checkerboard when the grid has an **odd** number of columns — this
     project's 8×9 (9 columns) is safe; a 9×8 layout would not be without
     reordering characters. This is checked in code
     (`validate_no_adjacent_co_flash()`), not assumed.
3. **Every flash pushes a marker** onto the `P300Markers` LSL stream at the
   moment of the screen flip (or, for the in-window board, at repaint-request
   time): a JSON object naming the trial, the flash index, which character
   indices are in this flash's group, and whether the current target is in
   that group (`is_target`). This per-flash detail — which group, not just
   target/non-target — is what lets any later analysis reconstruct exactly
   what the subject saw. The original reference implementation this project
   extends did *not* log this, making its data unusable for replay; this
   project logs it from every recording, always.

### 3.2 The recorder merges EEG and markers into one file

`collection/record_session.py` runs on the same deployment machine. It:

1. Resolves the `EEG` stream (retrying with exponential backoff — 0.5s, 1s,
   2s, 4s, then every 5s — for up to 30s before giving up).
2. Starts writing EEG to disk **immediately** once found — before the
   stimulus even starts — so no pre-stimulus baseline is lost.
3. Opens the marker inlet the instant the `Markers` stream is found (an LSL
   inlet buffers nothing before it's opened, so this has to happen right
   away, not lazily).
4. For every EEG sample, checks whether a marker landed closest to it in
   time and attaches that marker's JSON to that CSV row. A short 1-second
   "holdback" window means a marker that arrives slightly late (markers are
   timestamped at the screen flip, EEG can arrive a beat later) still gets
   matched to the correct sample rather than being dropped.
5. Detects and flags problems rather than silently swallowing them: stream
   dropouts (written as explicit gap rows with all-NaN channels), duplicate
   markers, non-monotonic timestamps, a rate that's drifted off 250 Hz, and
   trials whose `trial_start` never arrived.

The output is one CSV per session with columns:
```
timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, marker_json_or_empty,
trial_id, flash_idx, group_json, is_target, target_char_idx
```
plus two side files: `<csv>.log.jsonl` (a structured, line-by-line log of
everything the recorder noticed) and `<csv>.integrity.json` (an end-of-session
summary — gap counts, rate drift, trials missing a `trial_start`, etc.). A
demo/mock recording is always prefixed `demo_` so it can never be mistaken
for real data.

**Clock handling:** EEG timestamps are mapped into this machine's own clock
via LSL's `proc_clocksync` (a no-op correction on one machine, but it stays
on unconditionally so the split-machine variant, or any future multi-clock
setup, isn't silently broken). Deliberately **no dejitter** is applied —
dejitter was measured to silently shift timestamps up to ~19 ms specifically
to smooth over
non-monotonic steps, which is exactly the evidence the integrity checks need
to see. If the real Unicorn interface turns out to need it, there's a single
switch (`pipeline/constants.EEG_INLET_DEJITTER`) that the recorder and the
live inference service both read, so training and serving would stay
consistent with each other either way.

---

## 4. Calibration / fitting: turning a recording into a usable model

`pipeline/fit_from_recording.py` runs offline, after a calibration recording
exists:

```bash
python -m pipeline.fit_from_recording --recording collection/recordings/<file>.csv --id subject_001
```

1. **Load the CSV.** Rows where all 8 channels are finite numbers are EEG;
   everything else (gap rows, marker-only rows) is metadata. Flash onset
   time and its target/non-target label come straight from each flash
   marker row.
2. **Cut epochs** (`pipeline/epoching.py`, the exact same code the live
   service uses later, so training and serving see identical
   preprocessing): a window from flash onset to +0.8 s, with 0.5 s of extra
   padding on each side so a zero-phase 1-24 Hz Butterworth band-pass filter
   has room to settle before the window is cropped back down. No baseline
   correction by default. Everything stays in microvolts throughout.
3. **Train the classifier with honest cross-validation.** A fresh
   `xDAWNCovariances → TangentSpace → shrinkage LDA` pipeline
   (`classifiers/xdawn_cov_ts_regLDA.py`) is fit on each of several
   `GroupKFold` splits **grouped by trial_id** (so flashes from the same
   trial never leak across train/test), producing out-of-fold decision
   scores for every epoch. Those out-of-fold scores — never in-sample scores,
   which would be overconfident — are used to fit **Platt scaling**
   (`stopping/calibrate.py`): a simple sigmoid `(a, b)` that turns a raw,
   unitless classifier score into an actual probability `P(target | score)`.
   Expected Calibration Error (ECE) is measured on the same held-out scores.
4. **Fit the final classifier** on *all* epochs (now that cross-validation
   has already produced an honest score estimate) and save it.
5. **Write three files:**
   - the **classifier checkpoint** (`models/<id>_classifier.joblib`) — the
     fitted pipeline plus the exact epoch-cutting parameters it was trained
     with;
   - the **calibration file** (`models/<id>_platt.json`) — the Platt `(a,
     b)`, the ECE, and the sha256 of the checkpoint it belongs to (so it can
     never silently attach to the wrong classifier);
   - the **session manifest** (`session_manifest.json`) — the file every
     other process actually loads. It names the characters file, the
     classifier, the calibration, the language model, and the stopping
     parameters (θ, t_min, t_max, λ), each identified by a sha256 that's
     re-verified every time it's loaded, plus one overall fingerprint (a
     hash of everything together) that lets two processes confirm they're
     using byte-identical models without comparing file contents directly.

Safety rails: a recording named `demo_*` is refused unless `--allow-demo` is
passed (and if it is, every output file is also `demo_`-prefixed); an
existing output file is never silently overwritten without `--force`.

---

## 5. Live inference: from flash to scored evidence

`pipeline/inference_service.py` is a separate process, started by the GUI,
that consumes EEG and markers **directly from LSL** (not from the recorder's
CSV — lower latency, and no race against a file being written in chunks) and
publishes decisions on the `P300Inference` stream.

Per flash, in order:

1. **Wait for the epoch to exist.** A flash marker arrives; the service
   waits until enough EEG has actually accumulated in its rolling buffer to
   cover that flash's window (onset to +0.8 s, plus the 0.5 s padding the
   classifier's filter needs) before trying to cut it.
2. **Reject epochs that can't be trusted**, rather than silently degrading:
   - `epoch_incomplete` — the window spans a timestamp gap, non-monotonic
     timestamps, or the EEG simply never arrived (a flash queued for more
     than 5 s past when its window should have completed is rejected rather
     than blocking the trial forever).
   - `quality_gate` — any one of the 8 channels was red (railed) or unknown
     over that window. Because xDAWN spatially mixes *all* channels
     together, one bad electrode contaminates the whole epoch, so the
     default policy drops the entire flash rather than trying to salvage it.
     Yellow (noisy but not dead) never drops a flash.
   - A rejected flash contributes to neither the minimum nor the maximum
     flash count the stopping controller uses — it's as if that flash never
     happened.
3. **Score the epoch.** The saved classifier pipeline's `decision_function`
   turns the (channels × time) epoch into one real number; positive means
   "more target-like."
4. **Convert the score to a log-likelihood ratio and update the posterior.**
   This is the part deliberately left for the project owner to implement by
   hand (`stopping/bayesian_stopping.py` — `llr()` and `update()` are
   `NotImplementedError` stubs on purpose, to preserve the math as the
   project's own contribution rather than generated code). Conceptually:
   Platt scaling turns the raw score into `P(target | score)`; that becomes a
   log-likelihood ratio which is added, in the log domain, only to the
   entries of the character posterior belonging to this flash's group (the
   other characters weren't shown, so their evidence doesn't change); the
   whole posterior is renormalized; and the flash count `t` increments.
5. **Check the stopping condition** (also owner-implemented): once at least
   `t_min` flashes have been seen, stop as soon as the posterior's maximum
   probability crosses a fixed threshold `θ`; otherwise force a stop at
   `t_max` flashes regardless, taking whichever character currently has the
   highest posterior probability.
6. **Publish what happened.** Every flash's outcome becomes a
   `posterior_update` event (current leading character, its probability,
   θ); a decision to stop becomes a `character_committed` event; a dropped
   flash becomes a `flash_rejected` event. The service also publishes its
   own health as `service_status` events (`starting` → `ready`, repeated
   every 2 s as a heartbeat since LSL never replays missed samples to a late
   subscriber — so the GUI can connect at any time and still know within 2 s
   whether the service is alive; `warning` / `error` when something's wrong).

A **language model prior** feeds into step 4 at the start of each new
trial (`reset()`, also owner-implemented): today an n-gram over Thai
characters conditioned on what's already been typed this session, blended
with a uniform floor so the prior can never rule a character out entirely.
A Transformer-based prior is planned as a drop-in replacement for the same
slot later.

**A character never comes for free at t_max.** If the stimulus's fixed
number of repetitions runs out before either stopping condition fires, a
`trial_end` marker forces a commit anyway (`reason: trial_ended`) — but this
is reported as a *distinct* reason from an honest threshold-crossing, both to
the log and visually in the GUI, never blended into "it worked."

---

## 6. How the character display actually updates

The GUI (`collection/session_gui_qt.py`, logic in `collection/gui_logic.py`)
subscribes to the `P300Inference` stream as just another LSL consumer — it
never talks to the classifier or the controller directly. On every refresh
tick (100 ms):

1. It pulls any new `P300Inference` events and hands each to
   `InferenceDisplayModel`, a pure (no-Qt) state machine so the update logic
   is unit-testable without a GUI.
2. **Before trusting anything**, it checks the stream's fingerprint (carried
   in the LSL outlet's `desc()`) against the manifest the panel itself
   loaded. If they don't match — stale service, wrong session, crossed
   wires — every event from that stream is ignored and the mismatch is
   shown, rather than silently displaying decisions from the wrong model.
3. `posterior_update` events update a **confidence gauge**: which character
   currently leads, and how close its probability is to θ — so the operator
   can see evidence accumulating flash by flash, not just the final answer.
4. `character_committed` events append one character to the **composed
   text** shown on screen (`InferenceDisplayModel.composed_text` is just the
   concatenation of every committed character so far this session). The
   *reason* for the commit is tracked alongside it and shown visually
   distinct — never by color alone — when it was a fallback (`trial_ended`
   or `max_flashes_reached`) rather than an honest threshold crossing.
5. `flash_rejected` events increment visible counters (quality-gate
   rejections vs. everything else), so a run of bad flashes is visible to
   the operator in real time rather than only in a log file afterward.
6. A malformed or schema-invalid event is counted and shown, but never
   crashes the display — one bad event shouldn't take down a live session.

Separately, the **scalp map** (electrode-quality dots) is driven by the raw
EEG stream directly, not by inference events: each channel's peak-to-peak
amplitude over its rolling window is classified green (1-200 µV, good),
yellow (200-400 µV, noisy), or red (outside that range, or any 100 ms
segment under 1 µV — added specifically because whole-window peak-to-peak
alone let a contact that failed mid-window stay green for almost a second).
The exact same thresholding function feeds both the GUI's dots and the
inference service's quality gate, so the two can never disagree about
whether a channel was good enough to trust.

---

## 7. One flash, start to finish — the short version

```
Screen flashes a group of characters
        |  (P300Markers: {event: "flash", trial_id, flash_idx, group, is_target})
        v
record_session.py writes the marker + surrounding EEG samples to the CSV
        |  (separately, live:)
        v
inference_service.py buffers EEG, waits for the flash's epoch window to fill
        v
epoching.py cuts a 0-0.8s window, band-pass filtered
        v
quality.py gates the epoch (drop if any channel is red/unknown)
        v
classifier_io.py: fitted pipeline -> decision_function score
        v
stopping/bayesian_stopping.py: score -> Platt-scaled P(target) -> LLR
                                -> add to this flash's group's posterior entries
                                -> renormalize -> check theta / t_min / t_max
        v
P300Inference: posterior_update (every flash) or character_committed (on stop)
        v
session_gui_qt.py: confidence gauge updates, or a character is appended
                    to the composed text on screen
```

---

## 8. What's still a known limitation

- **The stimulus is open-loop.** It always runs its fixed number of
  repetitions; even after the controller commits a character early, the
  subject still sits through the rest of that trial's flashes before the
  next cue appears. The ITR (information transfer rate) benefit of stopping
  early isn't realized live yet — the service just ignores the trial's
  remaining flashes once it has committed.
- **The real headset's montage is verified, but its reported channel labels are not.** The 8-channel layout Fz/C3/Cz/C4/Pz/PO7/Oz/PO8 is confirmed as the Unicorn Hybrid Black's factory-default cap (g.tec's own user manual, cross-checked against a MathWorks listing and published papers using this headset — see `CLAUDE.md`'s hard constraints). What's still unconfirmed is whether the live stream *reports* those same label strings. The service refuses to start if the live stream's labeled electrodes don't match what the checkpoint was trained on, and warns rather than silently trusting generic labels like "EEG 1".
- **In-window board timing is estimated, not measured against the actual
  screen.** Qt on macOS has no presentation timestamp, so its flash markers
  are stamped when a repaint is *requested*, measured within ~7 ms of the
  planned schedule — but the further lag to the physical screen is
  unmeasured. The separate PsychoPy window (GPU-flip-timestamped) remains
  the precise option for real data collection until a photodiode check
  validates the in-window board.
- **`reset()`, `llr()`, and `update()` in `stopping/bayesian_stopping.py`
  are intentionally unimplemented.** This is not a bug — the project owner
  is implementing the Bayes posterior update, the score-to-LLR conversion,
  and the stopping-condition check by hand, deliberately, to keep that math
  as their own worked derivation.

## 9. Standing decisions (see `CLAUDE.md` for the full, authoritative list)

- **The inference service stays a strict checkpoint loader, never a fallback
  trainer.** It refuses to start (exit 2) unless the checkpoint's channel
  order, sample rate, and epoch window match the live stream exactly — this
  is why tests use a scripted/mocked classifier, and why real inference
  stays blocked until `pipeline/fit_from_recording.py` has produced a real
  checkpoint from an actual calibration recording. Fitting that checkpoint
  per subject is calibration, not "retraining the algorithm" — no algorithm
  redesign is implied, and fitted weights are never expected to transfer to
  new hardware without a new fit.
- **No dependency on g.tec's Unicorn Speller, for any reason** — not a
  fallback, not a comparison baseline, not a customization target. The
  stimulus, classifier, calibration, language-model prior, and stopping
  logic are fully custom and stay fully custom; the only g.tec software this
  project touches is the Unicorn LSL Interface, a raw data relay with no
  P300 logic of its own.
