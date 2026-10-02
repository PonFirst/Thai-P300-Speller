# Thai P300 Speller with Language-Model-Informed Bayesian Dynamic Stopping
## Research Dossier: every component, the math, the evidence, and a Q&A appendix

Prepared from the repository state of 2026-09-28. Every number in this document was read from
the code, measured, or computed from the code's own formulas; where a number is synthetic,
unverified, or still open, it says so next to the number.

---

## 0. Instructions for any agent using this document

This document is source material for building a presentation (research story, novelty, math)
and a Q&A appendix. Follow these rules. They exist because each one has been a real problem.

1. **No real-subject results exist yet.** No real calibration session has been fitted and no
   real online spelling session has been run. Never present any number in this document as
   "the system's accuracy" or "the system's ITR". Numbers come in these kinds only:
   - **[MEASURED-PUBLIC]**: classifier results on public datasets (BNCI2014-008/009), flash level.
   - **[MEASURED-LM]**: offline language-model comparisons (a 16-item test set, small).
   - **[MEASURED-HW]**: hardware/link measurements from early bench tests.
   - **[MEASURED-SW]**: software timing measurements on the development Mac.
   - **[SYNTHETIC]**: produced by the real code on scripted (fake) classifier scores. These
     demonstrate that the pipeline works. They are not research results and must be labelled
     as synthetic on any slide that shows them.
   - **[COMPUTED]**: arithmetic from a formula (e.g. ITR tables). Illustrative only.
2. **The paper's novelty angle has NOT been decided by the owner.** Section 2.5 lists
   candidate framings. Present them as candidates or pick one only if the owner tells you to.
   Do not invent a claim like "first ever" unless it is marked verified.
3. **Citations marked UNVERIFIED must be checked against the source before use.** This project
   has a documented history of AI-generated citations that turned out to be fabricated.
4. **The core stopping math (`reset`, `llr`, `update` in `stopping/bayesian_stopping.py`)
   was derived and implemented by hand by the project owner.** Credit it that way.
5. **The Mac hardware path is unproven** (see §4.5, §7). The Windows path uses g.tec's own
   relay software and has not been run end to end with a subject either.
6. **Quote numbers with their caveat.** Example: "81% top-1 on a 16-item test" is fine;
   "81% accurate" is not.

Tag legend used throughout: **[VERIFIED]** checked against a primary source or the code;
**[OPEN]** not yet known or not yet done; **[UNVERIFIED]** claimed but not checked.

---

## 1. Executive summary

- **What it is.** A brain-computer interface (BCI) that lets a person type Thai by attention
  alone. A 72-character Thai matrix flashes in groups; when the character the person is
  attending to flashes, the brain produces a P300 event-related potential (ERP) about 300 ms
  later. An 8-channel wireless EEG headset (g.tec Unicorn Hybrid Black, 250 Hz) records it,
  a classifier scores each flash, and the system decides which character was intended.
- **Baseline it extends.** Pananookooln et al. 2022 (AIT Brain Lab) built a Thai P300 speller
  with the checkerboard paradigm. This project reimplements that paradigm from scratch and
  extends it. [Details of the reference paper's own results: OPEN, verify from the paper.]
- **The research contribution (the part that is not reimplementation).**
  **Bayesian dynamic stopping with a Thai character-level language-model (LM) prior.**
  Instead of flashing every character a fixed number of times, the system keeps a probability
  distribution over all 72 characters, updates it after every single flash using calibrated
  classifier evidence, and stops as soon as one character is confident enough. Before any
  flash, the distribution starts from what a language model predicts the next character is,
  given the text typed so far.
- **The language model.** A 5.6M-parameter character-level GPT-style Transformer trained on the
  full Thai Wikipedia corpus (Colab GPU), whose output is marginalized down to the 72 matrix
  characters. An order-4 n-gram model and a uniform prior are kept as baselines.
- **Why it should help (the hypothesis).** Thai spelling has strong local structure: tone
  marks, vowel signs and common words make the next character highly predictable in many
  contexts (e.g. after "สวัส" the LM puts 99.4% on "ด"). A good prior lets confident
  characters stop after a handful of flashes while uncertain ones take longer, which should
  raise the information transfer rate (ITR) without lowering accuracy. **This is a hypothesis;
  it has not been tested on a real subject yet.**
- **Status.** Full pipeline built and tested (696 automated tests): stimulus, recorder, signal
  quality gate, epoching, classifier fitting with calibrated probabilities, live inference
  service, stopping controller, three LM types, experiment logging, and an automatic report
  generator. Public-dataset classifier validation done. Hardware connection partially
  demonstrated. **Real-subject data collection is the next step.**

---

## 2. The research story

### 2.1 What a P300 speller is

- **The P300.** When a person is watching for a rare, relevant stimulus among frequent
  irrelevant ones (the "oddball paradigm"), the brain produces a positive voltage deflection
  over centro-parietal scalp roughly 250-550 ms after the relevant stimulus. It is called the
  P300 because the classic peak is near 300 ms. This project's EDA searches for it in the
  window 0.25-0.55 s (`config.P300_SEARCH_WINDOW`). [VERIFIED: code]
- **The speller idea.** Show a grid of characters. Flash groups of them in rapid succession.
  The person silently counts flashes of the one character they want. Flashes that contain
  that character are "target" flashes and evoke a P300; all others are "non-target". By
  finding which flashes evoked a P300, the system infers the character.
- **Why it is hard.** A single-trial P300 is a few microvolts buried in background EEG that
  is several times larger. One flash is weak evidence. Classic spellers therefore repeat every
  flash a fixed number of times and average. That repetition is the main cost in time.
- **Speed metric.** Performance is reported as accuracy and as Wolpaw information transfer
  rate (ITR, bits/minute), which combines accuracy, the number of choices and time per
  selection (§5.13).

### 2.2 Why Thai, and what makes it different

- Thai script has 44 consonants, vowel signs written above, below, before or after the
  consonant, four tone marks, and other diacritics. Written Thai does not put spaces between
  words.
- A visible Thai syllable is often several code points. For example "สวัสดี" (hello) is typed
  as ส, ว, ั, ส, ด, ี: six selections, two of which are marks with no base of their own.
- The matrix therefore contains combining marks as their own cells. In a speller, each mark
  costs a full selection. This is the property that makes a language model valuable: marks
  and vowel signs are often nearly determined by context.
- Thai is under-studied in P300 spelling compared with English and Chinese. [Claim about
  literature coverage: UNVERIFIED, do a literature check before stating it.]

### 2.3 The problem this project addresses

1. **Fixed repetitions waste time.** The standard protocol flashes every group the same number
   of times (here 5 repetitions = 120 flashes per character) whether the evidence became
   decisive after 10 flashes or never did.
2. **The standard protocol ignores language.** The previous characters make the next one far
   more predictable, but a classic speller starts every character from zero.

### 2.4 The approach in one paragraph

Treat the choice of character as sequential Bayesian inference over 72 hypotheses. The prior
comes from a language model conditioned on the text typed so far, blended with a uniform
floor so no character is ever ruled out. Each flash's classifier score is converted to a
calibrated log-likelihood ratio (Platt scaling plus a group-size correction) and added to the
log-probabilities of exactly the characters in the flashed group. Stop when the most probable
character passes a fixed threshold θ after at least t_min flashes, or at a flash limit t_max.

### 2.5 Candidate contribution framings (owner has NOT chosen; present as options)

| Framing | Claim it would make | Evidence it needs that does not exist yet |
|---|---|---|
| A. Methods | LM-informed Bayesian dynamic stopping raises ITR over fixed repetitions at equal accuracy in a Thai P300 speller | Real-subject sessions; replay of the same recordings under fixed-rep vs dynamic and uniform vs n-gram vs Transformer priors |
| B. Language-specific | A character-level Transformer prior, marginalized to the speller's alphabet, is well suited to Thai's combining marks | Per-character-class analysis (consonants vs marks) of flashes saved, on real data |
| C. Systems / reproducibility | A fully open, vendor-independent pipeline (including a Mac relay speaking the headset's documented Bluetooth protocol), strict train/serve parity, hash-verified manifests | Mac path passing hardware checks H12-H14; independent replication |

Rejected idea (do not present as planned work): replacing some matrix cells with LM-predicted
whole-word completions. It was explored and rejected on its merits (§Appendix F.3); it may
appear only as future work in a discussion section.

### 2.6 Context: where this sits

PhD application portfolio work (targeting CMU ECE/BME). Per the older project notes
(`docs/claude.md`), this is the first project in an advisor-set sequence, chosen because it
works with existing 8-channel hardware, carries a light human-subjects burden, and is
unexplored for Thai. [Ethics/IRB approval status for data collection: OPEN, not recorded in
the repository. Do not claim approval.]

---

## 3. System architecture

### 3.1 End-to-end picture

```
 Subject wearing Unicorn Hybrid Black (8 EEG ch, 250 Hz, Bluetooth)
        |
        |  Bluetooth serial, 45-byte packets
        v
 Windows: g.tec Unicorn LSL Interface      OR      Mac: collection/unicorn_relay.py
        \____________________  ___________________/
                             \/
                LSL stream "UnicornEEG" (type EEG, 8 ch, float32, microvolts)
                             |
      +----------------------+-----------------------------+
      |                      |                             |
      v                      v                             v
 Qt control panel      record_session.py              inference_service.py
 (traces, scalp map,   (CSV: EEG + markers,           (epoch -> quality gate ->
  launches everything)  gap rows, integrity)            classifier score -> LLR ->
      ^                      ^                          Bayesian posterior -> stop?)
      |                      |                             |
      |        LSL "P300Markers" (JSON: trial_start / flash / trial_end)
      |                      |                             |
      |        Stimulus: PsychoPy window or in-panel Qt board (checkerboard flashes)
      |                                                    |
      +------------ LSL "P300Inference" (posterior_update, character_committed, ...) <--+
```

### 3.2 The four phases of use

1. **Calibration recording.** The subject copies a known sentence. Each cued character runs
   the full flash sequence. Output: a CSV with EEG and per-flash markers (which characters
   flashed, and whether the cued target was among them).
2. **Fitting** (`pipeline/fit_from_recording.py`). Cut epochs, train the classifier, calibrate
   its scores into probabilities with out-of-fold Platt scaling, write the classifier, the
   calibration, a fit report and a hash-verified session manifest that also names the LM.
3. **Online spelling.** The inference service loads the manifest and decodes live, character
   by character, with dynamic stopping and the LM prior.
4. **Evaluation.** Offline replay of recorded scores under swept hyperparameters
   (`stopping/evaluate_recording.py`), and per-session summaries/reports from the experiment
   logs (`stopping/summarize_experiment.py`).

### 3.3 Processes and how they talk

| Process | Role | Talks through |
|---|---|---|
| Unicorn LSL Interface (Windows) / `unicorn_relay.py` (Mac) | headset -> LSL | publishes `UnicornEEG` |
| `collection/record_session.py` | durable recording | reads `EEG` + `Markers` |
| stimulus (`checkerboard_stimulus.py` or `board_qt.py`) | shows flashes | publishes `P300Markers` |
| `pipeline/inference_service.py` | live decoding | reads `EEG` + `Markers`, publishes `P300Inference` |
| `collection/session_gui_qt.py` | operator panel | reads `EEG` + `P300Inference`, launches the others |

Every consumer finds streams **by type, never by host name**, so the same code runs on one
machine or across several. [VERIFIED: code]

---

## 4. Components in detail

### 4.1 Hardware: g.tec Unicorn Hybrid Black

| Property | Value | Source |
|---|---|---|
| Channels | 8 EEG + accelerometer, gyroscope, battery, sample counter | g.tec Bluetooth protocol document [VERIFIED] |
| Sampling rate | 250 Hz per channel | g.tec spec page; `EEG_SFREQ = 250.0` [VERIFIED] |
| ADC | 24-bit | g.tec spec page [VERIFIED] |
| Input range (sensitivity) | ±750 mV | g.tec spec page [VERIFIED] |
| Input impedance | > 1 GΩ | g.tec spec page [VERIFIED] |
| Resolution | 750,000 µV / 2^23 ≈ 0.0894 µV per count; protocol formula 4,500,000 / 50,331,642 = 0.089406 µV/count | both g.tec documents agree [VERIFIED] |
| Montage (channels 1-8) | Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 | Unicorn Hybrid Black user manual, cross-checked [VERIFIED]; the live stream's own reported labels: [OPEN, check H2/H13] |
| Reference / ground | sticky electrodes on both mastoids | g.tec quick start guide [VERIFIED] |
| Electrodes | hybrid (dry or with gel) | g.tec quick start guide |

- **Montage correction (important history).** The project first assumed the reference paper's
  cap (Fz, F3, C4, Cz, Pz, P3, O2, O1). That was never this hardware. Four of eight positions
  are genuinely different sites (C3 not F3, PO7 not P3, Oz not O2, PO8 not O1). Any recording
  made before the correction is treated as invalid. `pipeline/constants.EEG_CHANNELS` is the
  single definition.
- **Amplifier question from the advisor.** There is one fixed hardware amplifier/ADC with the
  spec above. The software applies exactly one scale factor (counts to µV, g.tec's own
  formula) and no other gain anywhere. The live trace display auto-scales each channel for
  visibility (§4.14); that is display-only and never touches recorded values.
- **LED.** Color = battery (cyan OK, yellow low, red empty, green charging). Blink pattern =
  state (flashing = on and not connected, slow blink = Bluetooth link open, solid = streaming).

### 4.2 The character matrix

`collection/characters_th.json`, 72 characters, laid out 8 rows × 9 columns, row-major.

| Group | Count | Characters |
|---|---|---|
| Consonants | 44 | ก ข ฃ ค ฅ ฆ ง จ ฉ ช ซ ฌ ญ ฎ ฏ ฐ ฑ ฒ ณ ด ต ถ ท ธ น บ ป ผ ฝ พ ฟ ภ ม ย ร ล ว ศ ษ ส ห ฬ อ ฮ |
| Vowel signs and others | 17 | ะ า ิ ี ั ึ ื ุ ู เ แ โ ำ ไ ใ ๆ ็ |
| Tone marks and diacritic | 5 | ่ ้ ๊ ๋ ์ |
| Digits | 6 | 0 1 2 3 4 5 |

- **Deliberate deviation from the source list.** Digits 6 and 7 were dropped and ั (mai
  han-akat) and ็ (mai taikhu) added. Both marks are common (วัน, รัก, เด็ก, เย็น) and their
  absence forced unnatural calibration sentences. The count stays 72, so the checkerboard
  math is unchanged.
- **Not in the matrix:** space, ฤ, ฦ, ฯ, ๅ, ํ, ฺ, digits 6-9. So the speller cannot type a space
  or these rarer characters. [Design consequence, state it if asked.]
- **Status:** the list is marked `"verified": false` against the reference implementation's
  source file (`th_offline_cb.py`). [OPEN: confirm before real collection.]
- **Orientation:** 8×9 is adjacency-safe; 9×8 is not without reordering (54 violations
  measured). [VERIFIED: computed]

### 4.3 Stimulus paradigm: the checkerboard

Implementation: `collection/flash_sequence.py` (pure logic), shown by
`collection/checkerboard_stimulus.py` (PsychoPy) or `collection/board_qt.py` (in-panel Qt).

**Grouping (one repetition).**
1. Split the 72 characters into two colors by flat index parity: even = white (36), odd =
   black (36). With 9 columns (odd), index parity equals true 2-D checkerboard parity
   ((row + col) mod 2), so no two horizontally or vertically adjacent characters share a
   color. [VERIFIED: computed, 0 violations for 8×9]
2. Shuffle each color and reshape it into a 6×6 **virtual matrix**.
3. Flash, in order: the 6 rows of the white virtual matrix, the 6 rows of the black one, the
   6 columns of the white one, the 6 columns of the black one. **24 flashes per repetition,
   6 characters per flash.**
4. Reshuffle for every repetition.

**Properties that matter for the math and for Q&A.** [VERIFIED: computed]
- Each character appears in exactly **2 of the 24 flashes** per repetition (one row group,
  one column group of its own color), and those two groups intersect **only** in that
  character.
- The target-flash base rate is therefore **2/24 = 1/12**, and the group prior used in the
  math is **π = 6/72 = 1/12**. These are the same number (§5.5).
- **Adjacent characters never flash together**, which reduces adjacency errors (a known
  failure mode of row/column spellers).
- **Minimum spacing between two flashes of the same character is 7 flashes** (checked over
  200 random schedules × 72 characters) = 868 ms at the nominal 124 ms SOA, which is longer
  than the 800 ms epoch. So a target epoch never contains a second flash of the same target,
  avoiding the "double flash" overlap and repetition-blindness problems.

**Timing (defaults).** [VERIFIED: code]

| Parameter | Default | Meaning |
|---|---|---|
| Flash duration | 62 ms | group lit |
| ISI | 62 ms | gap before next flash |
| SOA | 124 ms | flash-to-flash interval |
| Repetitions | 5 | 120 flashes per character at most |
| Cue | 2000 ms | target highlighted before flashing (`trial_start` marker) |
| Inter-selection pause | 3500 ms | after a character |
| Fixed-protocol time per character | 2.0 + 14.88 + 3.5 = 20.38 s | [COMPUTED] |

- **Measured display timing on the M3 Max** (from the owner's research notes, measured in a
  separate session with software flip timestamps, no photodiode; not re-verified here): the PsychoPy window locks to 8.33 ms frames, so 62 ms rounds up to
  66.7 ms and the real SOA is **133.3 ms**, not 124 ms. The in-panel Qt board keeps 124 ms
  but stamps markers at repaint request (screen lag unmeasured). Consequences: calibrate and
  run online on the same board; offline evaluation measures seconds-per-flash from the
  recording's own markers, so reported ITR uses real timing. Flip-to-light lag: [OPEN, H7].
- **Markers.** One JSON object per LSL sample, stamped at the display flip:
  `{"event":"trial_start","trial_id","target_char_idx","target_char"}`,
  `{"event":"flash","trial_id","flash_idx","group":[6 indices],"is_target":bool}`,
  `{"event":"trial_end","trial_id"}`.
- **Cued (copy-spelling) sessions only.** Every trial cues a known target. This is what makes
  accuracy scorable. Free spelling (no cue) is not implemented.
- **Open-loop limitation.** The stimulus always runs its full repetitions. When the controller
  stops early, the service commits the character and ignores the remaining flashes, but the
  subject still sits through them. The ITR gain of dynamic stopping is therefore not yet
  realized live (§8). [OPEN]

### 4.4 Data transport: Lab Streaming Layer (LSL)

| Stream | Type | Format | Producer |
|---|---|---|---|
| `UnicornEEG` | `EEG` | 8 ch float32 µV, 250 Hz | Unicorn LSL Interface or `unicorn_relay.py` |
| `UnicornAux` | `Aux` | 8 ch float64: battery %, acc x y z (g), gyro x y z (°/s), counter | `unicorn_relay.py` only |
| `P300Markers` | `Markers` | 1 ch string JSON, irregular | stimulus |
| `P300Inference` | `P300Inference` | 1 ch string JSON, irregular | inference service |

Lessons encoded in the code (each was a measured failure first):
- An inlet buffers nothing until `open_stream()` is called (measured 0/10 markers without it).
- LSL never replays samples to a late subscriber, so the stimulus waits up to 15 s for a
  marker consumer plus 2 s, and the service re-sends its `ready` status every 2 s.
- Clock synchronization on, **dejitter off**: measured, dejitter silently moved timestamps by
  up to 18.7 ms to hide out-of-order stamps. One switch (`EEG_INLET_DEJITTER`) governs both
  the recorder and the service so training and serving stay consistent.
- `P300Inference` is deliberately not type `Markers`, or consumers would latch onto it.
- Local LSL latency on the dev Mac: push-to-pull median 0.099 ms, p99 0.172 ms. [MEASURED-SW]

### 4.5 The Mac path: a relay speaking g.tec's Bluetooth protocol

Why it exists: g.tec's software is Windows-only and BrainFlow's Unicorn support is compiled
out on macOS. The relay (`collection/unicorn_relay.py`, pure logic in
`collection/unicorn_protocol.py`) reads the headset's Bluetooth serial port (`/dev/cu.UN-*`)
and publishes the same `UnicornEEG` stream. No P300 logic; a stand-in for g.tec's relay.

**Protocol** (from g.tec's UnicornBluetoothProtocol.pdf, manual v1.18.00) [VERIFIED]:
- Start acquisition `61 7C 87`, stop `63 5C C5`, each acknowledged with `00 00 00`.
- 45-byte payload at 250 Hz:

| Bytes | Field | Encoding | Conversion |
|---|---|---|---|
| 0-1 | header | `C0 00` | |
| 2 | battery | low nibble v | % = 100·v/15 |
| 3-26 | 8 EEG channels | 3 bytes each, **big-endian** two's complement | µV = value · 4,500,000 / 50,331,642 |
| 27-32 | accelerometer x y z | int16 **little-endian** | g = value / 4096 |
| 33-38 | gyroscope x y z | int16 little-endian | °/s = value / 32.8 |
| 39-42 | sample counter | int32 little-endian | |
| 43-44 | footer | `0D 0A` | |

- The decoder reproduces g.tec's worked example exactly (CH1 = 3654.87 µV, counter 176).
- g.tec's document has a typo: its counter formula prints `Counter[1]` three times; its own C
  example (and this code) use bytes 0-3.

**Timestamps from the device counter, not Bluetooth arrival** (`CounterClock`):
- Bluetooth delivers packets in bursts, so stamping on arrival would smear every ERP.
- timestamp = counter/250 + offset. Bluetooth only adds delay, so arrival − counter/250 ≥ true
  offset; the minimum over a 15 s window estimates it. The offset is slew-limited to 200 ppm,
  which tracks crystal drift while keeping stamps strictly increasing. First 2 s held back to
  anchor the clock.
- Simulated result: stamp error std about 0.05-0.1 ms versus about 5.1 ms when stamping on
  arrival; max error 0.2-2.2 ms for drift up to 150 ppm. [SYNTHETIC simulation of the clock]
- A lost packet appears as a timestamp jump of (missing+1)×4 ms, which the recorder and
  epoching already detect.
- Relay CPU cost: 6.7 µs median per packet (p99 8.3 µs), 0.17% of the 4 ms budget. [MEASURED-SW]

**Hardware results so far** [MEASURED-HW, early bench tests, headset not worn]:
- First real connection: 2,298 packets, 0 lost, 0 resyncs; panel showed "EEG connected".
- A longer run: 1,120 of 22,378 samples lost (5.0%), in two bursts (about 14 gaps over ~6 s,
  then ~50 s clean, then 5 gaps), 0 resyncs. Pattern consistent with 2.4 GHz interference
  (the Mac's Bluetooth shares its antenna with Wi-Fi). [Cause: not proven]
- Later attempts: the link could not be established (0 bytes after the start command, with
  plain pyserial too; macOS showed the headset in range but "Not Connected"). Unresolved at
  the time of writing. [OPEN]
- The 30-minute sample-counter gap test (H12), montage check (H13) and timing check (H14)
  have not passed. Treat the Mac path as unproven.

### 4.6 Recorder

`collection/record_session.py`: one CSV per session with columns
`timestamp, Fz, C3, Cz, C4, Pz, PO7, Oz, PO8, marker_json_or_empty, trial_id, flash_idx,
group_json, is_target, target_char_idx`. Every flash row carries its full group and label.
- Gap rows (all channels NaN) mark stream gaps; markers that cannot attach to a sample are
  written as their own rows. Nothing is silently dropped.
- Sidecars: `<csv>.log.jsonl` (structured log) and `<csv>.integrity.json` (gaps,
  non-monotonic stamps, rate drift, duplicate/orphan markers, trials without `trial_start`).
- Exponential backoff to find the EEG stream; records EEG first, opens the marker inlet the
  moment it resolves (fixed an earlier 2.7 s marker loss).

### 4.7 Signal quality and the quality gate

`pipeline/quality.py`, shared by the panel's scalp map and the service's gate so they can
never disagree. Peak-to-peak (µV) per channel:

| ptp | Level |
|---|---|
| < 1 | red (flat or disconnected) |
| 1 to 200 | green |
| 200 to 400 | yellow (noisy) |
| > 400 | red (railed or clipped) |
| non-finite | unknown |

- Also red if **any 100 ms segment** is below 1 µV (a contact that fails mid-window shows red
  after 100 ms instead of about a second later).
- **Gate policy "skip" (default):** reject the whole flash if any channel is red or unknown.
  Reason: xDAWN mixes all channels, so one bad electrode contaminates every spatial component
  and would push a large, confident, wrong score into the posterior. Yellow never rejects
  (blinks routinely exceed 200 µV frontally). A rejected flash counts toward neither t_min
  nor t_max. Policy "warn" accepts everything, for offline comparison.

### 4.8 Epoching and preprocessing (train/serve parity)

`pipeline/epoching.py` is the single implementation used by both the fit script and the live
service, so a classifier is never served epochs cut differently from those it was trained on.

| Parameter | Value |
|---|---|
| Window | 0 to 0.8 s after flash onset (200 samples) |
| Band-pass | 1-24 Hz, 4th-order Butterworth, zero-phase (`sosfiltfilt`) |
| Padding | 0.5 s each side, filtered then cropped (removes edge effects) |
| Baseline correction | none on this path (the MOABB EDA path uses −0.2 to 0 s) |
| Units | µV, enforced |
| Rejection reasons | not_yet_available, before_buffer, gap_in_window, non_monotonic, non_finite |

- Filtering uses only the padded window, so an epoch is identical whether cut from a whole
  recording or from a live buffer (tested).
- A sample step larger than 1.5 × the sample period inside the window counts as a gap.

### 4.9 The classifier: xDAWN covariances + tangent space + shrinkage LDA

`classifiers/xdawn_cov_ts_regLDA.py`, used unchanged ("validated; fit it, don't change it").

```
epoch (8 × 200)
  -> XdawnCovariances(nfilter=4, estimator="scm")   # spatial filters + augmented covariance
  -> TangentSpace(metric="riemann")                 # SPD matrix -> flat vector
  -> LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")   # Ledoit-Wolf shrinkage
  -> decision_function -> one real score (positive = target-like)
```

Why each step (math in §5.12):
- **xDAWN** learns spatial filters that maximize the signal-to-signal-plus-noise ratio of the
  target ERP. Covariances are built on the filtered signal augmented with the class prototype
  responses, so they carry both waveform and spatial information.
- **Tangent space.** Covariance matrices live on the curved manifold of symmetric positive
  definite matrices. Projecting onto the tangent plane at their Riemannian mean gives
  ordinary vectors a linear classifier can use.
- **Shrinkage LDA.** The tangent vectors have many dimensions relative to the number of
  trials in a fold; Ledoit-Wolf shrinkage keeps the covariance estimate well conditioned.

**Public-dataset validation** [MEASURED-PUBLIC; flash-level; subjects 1-5; GroupKFold by
subject; AUC / balanced accuracy]:

| Pipeline | BNCI2014-008 (8 ch, ALS) | BNCI2014-009 (16 ch, healthy) |
|---|---|---|
| xDAWNcov+FgMDM | 0.703 / 0.627 | 0.824 / 0.698 |
| xDAWNcov+TS+RegLDA (used) | 0.703 / 0.548 | 0.824 / 0.630 |
| xDAWNcov+MDM | 0.646 / 0.574 | 0.835 / 0.709 |
| xDAWNcov+MeanField | 0.640 / 0.560 | 0.835 / 0.714 |
| xDAWN+LR | 0.679 / 0.631 | 0.718 / 0.621 |

Caveats to state with this table:
- Flash-level only: these datasets expose no row/column identity, so no character accuracy
  or ITR can be computed from them.
- The best pipeline differs by dataset; do not average it away.
- FgMDM and TS+RegLDA have identical AUC by construction (pyriemann 0.12's FGDA runs the same
  shrinkage LDA on the same features); their balanced accuracy differs through the decision
  threshold.
- An earlier run reported FgMDM 0.769 / 0.694 on 008; the two configurations have not been
  reconciled. Do not cite either as final.
- Subject 4 was the worst fold for every pipeline: a finding, not noise.
- Report AUC and balanced accuracy, not raw accuracy: class imbalance is about 5:1 on BNCI and
  11:1 in this project's paradigm (2 target flashes of 24).

### 4.10 Fitting and calibration (`pipeline/fit_from_recording.py`)

1. Load the recording; keep rows with all 8 channels finite; read flash markers (onset = the
   marker row's timestamp, label = `is_target`).
2. Cut epochs with the shared `EpochSpec`; count rejections per reason.
3. Refuse if too little data: at least 3 trials, 10 target and 10 non-target epochs.
4. **Out-of-fold scores:** GroupKFold grouped by `trial_id` (default 5 folds, capped at the
   number of trials), a fresh pipeline per fold. Every epoch is scored by a model that never
   saw its trial.
5. **Platt scaling on the out-of-fold scores only** (in-sample scores are overconfident and
   would inflate every likelihood ratio). Output: slope a, intercept b.
6. Expected calibration error (ECE, 10 equal-width bins) on the same out-of-fold scores.
7. Fit the final classifier on all epochs; save checkpoint (with its `EpochSpec` and library
   versions), calibration JSON (recording the classifier's sha256), fit report, and manifest.
- Safety: a `demo_` recording is refused unless `--allow-demo`, and then every artifact is
  `demo_`-prefixed; nothing is overwritten without `--force`; a median |sample| below
  0.001 is refused as "volts, not microvolts".
- LM selection happens here: `--lm-transformer-checkpoint` (Transformer), `--lm-corpus`
  (n-gram), or neither (uniform). Mutually exclusive.

### 4.11 The stopping controller (summary; full math in §5)

`stopping/bayesian_stopping.py`, `BayesianStoppingController` (owner-implemented):

| Method | What it does |
|---|---|
| `reset(lm_prior)` | once per character: p0 = λ·prior + (1−λ)/72, store log p0, t = 0 |
| `llr(s, π)` | (a·s + b) + ln((1−π)/π), clipped to ±8 |
| `update(group, s, π)` | add llr to the group's log-probabilities, renormalize, t += 1, check stop |

Defaults [VERIFIED: code]: λ = 0.7, θ = 0.90, t_min = 6, t_max = 170, clip = 8.0.
Verified against a hand-worked 4-character example in `tests/test_bayesian_stopping.py`
(uniform prior, group {0,1}, LLR = ln 3 gives posterior [0.375, 0.375, 0.125, 0.125]).

Note: with 5 repetitions (120 flashes) < t_max (170), the forced stop at t_max cannot fire
with defaults; if θ is never reached, the trial ends and the service commits the argmax with
the distinct reason `trial_ended`. Fallback commits are always labelled and shown differently
in the panel, never mixed with confident ones.

### 4.12 The language models

All three expose `get_prior(context) -> 72 probabilities summing to 1`, aligned to
`characters_th.json`, validated at the boundary (shape, finite, non-negative, sum within
1e-6). Selected in the session manifest; built by `pipeline/language_model.py`.

**(a) Uniform.** 1/72 each. The no-LM control.

**(b) Order-4 character n-gram** (`stopping/lm_prior.py`). Counts continuations of contexts of
length 0-3 (only matrix characters are predicted; any character may appear in the context).
Add-α smoothing (α = 0.1), then a backoff blend with uniform (math in §5.11). Fit at startup
from a corpus file (15.8 s on the 25M-character sample).

**(c) Character-level Transformer** (`stopping/transformer_lm.py`, trained in
`stopping/gpt_dev.ipynb`). The production choice.

| Item | Value |
|---|---|
| Architecture | decoder-only GPT (Karpathy "Let's build GPT" structure): token + learned positional embeddings, 6 pre-norm blocks (8-head causal self-attention, 4× ReLU MLP), final LayerNorm, linear head |
| Size | 5,602,941 parameters; n_embd 256, 8 heads (head size 32), 6 layers, dropout 0.1 |
| Context (block size) | 64 characters (chosen for train/serve parity: real sessions cap near 40 characters) |
| Data | `pythainlp/thai-wiki-dataset-v3` (Thai Wikipedia, Wikiquote, Wikibooks, Wikisource, Wiktionary), all 3 parquet shards, about 430 MB, 196,533 articles, CC-BY-SA-3.0; articles shuffled (seed 1337); 90/10 train/val split |
| Vocabulary | 1,661 tokens: every character with ≥ 100 occurrences, plus all 72 matrix characters force-included, plus `<unk>` = U+E000 |
| Training | AdamW, lr 3e-4, batch 64, 20,000 steps (random windows with replacement, ≈ 82M positions), Colab T4 GPU, 1,545 s (≈ 26 min) |
| Final loss (cross-entropy, nats/char) | train 1.535, val 1.163 (val below train from step 1,000 onward; no sign of overfitting) |
| Checkpoint | `models/thai_gpt_checkpoint.pt` (22 MB, gitignored) |

- **Force-include fix.** ฃ and ฅ are archaic but in the matrix; they occur only 19 and 33
  times in a 25M-character sample, below the cutoff. Without force-inclusion they would merge
  into `<unk>` and the prior for those cells would be meaningless.
- **Marginalization.** The model predicts over its full 1,661-token vocabulary (it learned
  real, unrestricted Thai), then the 72 matrix probabilities are extracted, 1e-8 added, and
  renormalized (§5.3). Same convention as the n-gram.
- **Stale-checkpoint guard.** A checkpoint is only valid for the matrix it was trained with.
  Loading raises an error if any matrix character is missing from its vocabulary (no silent
  `<unk>` fallback, which would give a valid-looking but wrong distribution).
- **Training location.** Always on Colab, never on the development laptop (owner's rule).
- **First pass (superseded, kept for comparison):** 878,867 params, block 128, 128-dim, 4
  heads, 4 layers, 25M-character sample, 5,000 steps on the laptop GPU (155 s), val ≈ 2.01.

**LM comparison** [MEASURED-LM; 16 hand-picked common Thai completions, e.g. สวัส→ด, ขอบคุ→ณ;
CPU on an M3 Max]:

| LM | Top-1 | Top-3 | Mean P(correct) | Mean surprisal | Mean `get_prior` | One-time cost |
|---|---|---|---|---|---|---|
| Transformer, full corpus | 81.3% | 100% | 0.776 | 0.63 bits | 6.7 ms | 46 ms load |
| N-gram, order 4 | 75.0% | 100% | 0.630 | 0.96 bits | 0.008 ms | 15.8 s fit |
| Transformer, first pass | 68.8% | 93.8% | 0.639 | 1.18 bits | 2.0 ms | 10 ms load |

- Sixteen items: the ranking is suggestive, not a publishable evaluation. The proper
  comparison is replaying the same real recording under each LM (§6).
- Example predictions (Transformer): "ประเทศไ"→ท 0.93; "สวัส"→ด 0.99; "ขอบ"→ spread
  (เ 0.21, ค 0.11, ไ 0.09): appropriately uncertain where several words fit.
- **Latency is not a constraint.** `get_prior` runs once per character at `trial_start`, not
  per flash. The budget is the 2,000 ms cue; 6.7 ms is about 300× under it on the Mac.
  [Windows machine: OPEN, unmeasured]

### 4.13 The live inference service (`pipeline/inference_service.py`)

- `InferenceEngine` is pure logic (no LSL, threads or clock), fully testable; `ServiceRunner`
  does the LSL wiring.
- **Strict checkpoint loader, never a fallback trainer.** It reads everything from the
  manifest and refuses to start (exit 2) unless the checkpoint's channel order, sample rate
  and epoch window match the live stream. Electrode labels that differ from the checkpoint are
  a montage mismatch (exit 2); generic labels ("EEG 1") produce a warning.
- Per flash: wait until the epoch window is buffered → reject if incomplete (or stale for
  > 5 s) → quality gate → classifier score → `update()` → publish `posterior_update`; on stop,
  `character_committed`.
- Events (strictly validated both ends): `posterior_update`, `character_committed` (reason
  threshold_crossed / max_flashes_reached / trial_ended), `flash_rejected`, `service_status`
  (starting / ready / warning / error / stopped, with the manifest fingerprint).
- The cued target is read from `trial_start` **only to score commits in the log**; decoding
  never uses it.
- Latency: a flash's decision can only be made after its epoch completes (≈ 0.8 s + 0.5 s
  padding ≈ 1.3 s after the flash). Everything else is milliseconds.
- Exit codes: 0 stopped, 1 crash, 2 configuration, 3 no EEG, 4 controller not implemented.

### 4.14 The operator panel (`collection/session_gui_qt.py`, logic in `gui_logic.py`)

- Live traces, scalp map with quality colors, target entry, settings, composed text,
  confidence gauge, commit log. Launches service → recorder → stimulus in order and waits for
  each to report ready (service 20 s, recorder 35 s).
- **Trace display is per-channel auto-normalized:** (x − mean) / (4·std) + offset. It shows
  shape, not amplitude; a 2 µV noise channel and a 200 µV channel fill the same height. The
  scalp map colors use real µV thresholds and are the trustworthy quality indicator. (A
  fixed-µV display option has been offered, not built.) [This is why the advisor said the
  plot "looks weird".]
- Demo mode (`./start_speller.sh --demo`) runs a synthetic headset; all its outputs are
  `demo_`-prefixed and bannered.

### 4.15 Reproducibility: the session manifest (`pipeline/manifest.py`)

`session_manifest.json` names every artifact a run uses: characters file, classifier,
calibration, LM (uniform / ngram + corpus / transformer + checkpoint), stopping parameters,
quality-gate policy. Every artifact carries its sha256 and is re-verified at load (a file
swapped behind the same path is an error). The calibration file records the classifier hash
it was fitted on. The manifest's fingerprint is the sha256 of its canonical JSON, so two
processes reporting the same fingerprint are provably using byte-identical models.

### 4.16 Experiment logging and reporting

All files sit next to the recording (`pipeline/experiment_log.py`):

| File | Contents |
|---|---|
| `<csv>.session.json` | calibration/target sentence as typed, per-trial target list, settings, manifest, start/end, how far the session got |
| `<csv>.log.jsonl`, `<csv>.integrity.json` | recorder log and integrity summary |
| `<csv>.inference.log.jsonl` | `service_config` (models used); per character `trial_start` with the **LM prior** and the blended starting posterior; `flash_update` per flash (score, group, π, posterior max, target probability); `flash_rejected`; `char_committed` (reason, flashes, final posterior, correct) |
| `models/<id>_fit_report.json` | recording hash, epochs, per-fold and out-of-fold AUC, Platt a/b, ECE, versions, console output |

`stopping/summarize_experiment.py` turns these into accuracy with a 95% Wilson interval,
Wolpaw ITR, LM effect (probability given to the true character, rank-1 rate, flashes when
ranked 1st vs not), stop reasons, and a per-character table; `--html` writes a
self-contained report with charts, auto-generated findings and a benchmark comparison.

### 4.17 Testing and quality assurance

- **696 automated tests pass, 3 skip** (the skips check stub-only behaviour that no longer
  applies). No hardware needed; real-LSL tests use randomized stream types.
- Cross-component fault injection (`tests/test_fault_injection.py`): streams that never appear,
  drop mid-session, channels that go flat or rail, NaN classifier scores, t_max without
  threshold, manifest mismatch between processes, demo/real file collisions.
- Relay tests against a protocol-accurate fake headset, a pseudo-terminal and the real
  recorder, including garbage bytes, dropped packets and link loss.
- Hardware checks H1-H14 in `TESTING.md` (Appendix H) cover what software tests cannot.

---

## 5. The mathematics

Equations are numbered for slides and Q&A. Code references show where each one lives.

### 5.1 Notation

| Symbol | Meaning |
|---|---|
| 𝒞, N | the matrix characters, N = 72 |
| c* | the character the subject is attending to (unknown) |
| h | the text committed so far this session (LM context) |
| t | flash counter within one character (accepted flashes only) |
| G_t ⊂ 𝒞 | characters lit at flash t, \|G_t\| = 6 |
| y_t | 1 if c* ∈ G_t (a target flash), else 0 |
| s_t | classifier score for flash t (LDA decision function) |
| π | prior probability that a flashed group contains the target, π = \|G_t\|/N = 1/12 |
| q(c \| h) | LM prior over matrix characters |
| λ | LM blend weight (0.7) |
| p_t(c) | posterior P(c* = c \| h, s_1..s_t, G_1..G_t); ℓ_t(c) = ln p_t(c) |
| a, b | Platt slope and intercept |
| θ, t_min, t_max | stopping threshold (0.90), minimum flashes (6), maximum flashes (170) |

### 5.2 The problem as sequential Bayesian inference

Selecting one character is inference over 72 hypotheses {c* = c}. Evidence arrives one flash
at a time; after each flash the system decides whether to commit or keep flashing. The goal is
to commit as early as possible while keeping the probability of being wrong below 1 − θ.

### 5.3 The prior

**Eq. 1, LM marginalization** (`stopping/transformer_lm.py`, `get_prior`). The Transformer
outputs a distribution over its full vocabulary 𝒱 (|𝒱| = 1,661). The prior over the matrix is

$$q(c \mid h) = \frac{P_{LM}(c \mid h) + \varepsilon}{\sum_{c' \in \mathcal{C}} \big(P_{LM}(c' \mid h) + \varepsilon\big)}, \qquad \varepsilon = 10^{-8}$$

Apart from ε, this is exactly the conditional distribution P(next = c | h, next ∈ 𝒞): the
model is conditioned on the event that the next character is typeable. It is principled
conditioning, not a heuristic. The context is truncated to the last 64 characters (the
model's block size); an empty context feeds the single `<unk>` token.

**Eq. 2, blend with a uniform floor** (`reset`, owner-implemented):

$$p_0(c) = \lambda\, q(c \mid h) + (1-\lambda)\,\frac{1}{N}, \qquad \ell_0(c) = \ln p_0(c)$$

Properties (λ = 0.7, N = 72):
- **Floor:** p_0(c) ≥ (1−λ)/N = 0.00417. No character can be ruled out by the LM, so an LM
  mistake costs extra flashes, never an impossible character.
- **Ceiling:** p_0(c) ≤ λ + (1−λ)/N = 0.704 < θ = 0.90. The LM alone can never trigger a
  stop; EEG evidence is always required. (Example: an LM probability of 0.9945 becomes a
  starting posterior of 0.700.)
- With a uniform q, λ has no effect (p_0 is uniform either way).
- λ is a mixture weight expressing trust in the LM; it is fixed, not tuned. [Tuning λ on
  real data: OPEN]

### 5.4 The flash likelihood and the group-only update

**Assumption A1 (conditional independence).** Given c*, flash scores are independent, and a
score depends on c* only through whether c* was in the flashed group:

$$p(s_t \mid c^* = c, G_t) = \begin{cases} f_1(s_t) & c \in G_t \\ f_0(s_t) & c \notin G_t \end{cases}$$

**Eq. 3, Bayes' rule.** p_t(c) ∝ p_{t−1}(c) · p(s_t | c* = c, G_t). Divide every hypothesis
by the common factor f_0(s_t), which cancels on normalization:

$$p_t(c) \propto p_{t-1}(c)\cdot \Lambda(s_t)^{\mathbb{1}[c \in G_t]}, \qquad \Lambda(s) = \frac{f_1(s)}{f_0(s)}$$

**Eq. 4, log domain** (`update`, owner-implemented):

$$\tilde\ell_t(c) = \ell_{t-1}(c) + L(s_t)\,\mathbb{1}[c \in G_t], \qquad \ell_t = \tilde\ell_t - \operatorname{logsumexp}(\tilde\ell_t)$$

with L = ln Λ. Under A1 this is **exact Bayes, not an approximation**. Only the 6 flashed
entries change before normalization; the other 66 shift only through the normalizer.
Multiplication of probabilities becomes addition of log-probabilities.

**Eq. 5, log-sum-exp** (`log_normalize`): logsumexp(x) = m + ln Σ_i exp(x_i − m), m = max_i x_i.
Numerically stable for extreme values (tested at ±50).

### 5.5 From a classifier score to a log-likelihood ratio

**Eq. 6, Platt scaling** (`stopping/calibrate.py`), fitted on out-of-fold calibration scores:

$$P(y=1 \mid s) = \sigma(a s + b) = \frac{1}{1 + e^{-(a s + b)}}$$

**Eq. 7, remove the base rate.** Bayes at the flash level gives
logit P(y=1 | s) = ln Λ(s) + ln[π_cal / (1 − π_cal)], where π_cal is the fraction of target
flashes in the calibration data. Therefore

$$L(s) = (a s + b) + \ln\frac{1-\pi}{\pi}$$

(`llr`, owner-implemented). The code uses π = |G_t|/N = 6/72. In this paradigm the
calibration base rate is also 2/24 = 1/12 (each character is in 2 of 24 flashes), so
π = π_cal and the correction ln 11 = 2.398 is exact. [Subtlety for Q&A: if a different paradigm
or heavy epoch rejection changed the calibration base rate, the correction should use the
calibration set's actual rate. The calibration file records n_targets and n_scores, so this
is checkable.]

**Eq. 8, clipping:** L ← clip(L, −8, 8). e^8 ≈ 2,981, so one flash can change the odds of a
group by at most that factor. This limits the damage from artifacts, outlier scores and
violations of A1.

**Intuition, with illustrative Platt values a = 1.2, b = −2.4** [COMPUTED]:

| Score s | a·s + b | Calibrated P(target \| s) | L(s) | Likelihood ratio |
|---|---|---|---|---|
| +4.0 | +2.40 | 0.917 | +4.80 | 121 |
| +2.0 | 0.00 | 0.500 | +2.40 | 11 |
| 0.0 | −2.40 | 0.083 (= base rate) | 0.00 | 1.0 (no information) |
| −1.0 | −3.60 | 0.027 | −1.20 | 0.30 |

A 50% calibrated probability is strong evidence (×11) because the base rate is only 1/12.

### 5.6 The stopping rule

**Eq. 9** (`update`, owner-implemented). After each accepted flash:

$$\text{stop if } \big(t \ge t_{min} \ \text{and}\ \max_c p_t(c) \ge \theta\big)\ \ \text{or}\ \ t \ge t_{max}; \qquad \hat c = \arg\max_c p_t(c)$$

- θ (0.90) is the minimum posterior probability of being right at the moment of commitment.
  If the posterior is well calibrated, threshold stops should be correct at least θ of the
  time. [Whether the posterior is calibrated on real data: OPEN, measurable from the logs]
- t_min (6) stops a strong prior or one spurious flash from committing immediately.
- t_max (170) bounds time. With 120 flashes available it never fires; the stimulus's
  `trial_end` produces a separately labelled fallback commit instead.
- A fixed posterior threshold over many hypotheses is a relative of Wald's sequential
  probability ratio test. A boundary-aware (adaptive) threshold was considered and descoped.

### 5.7 Worked example [COMPUTED, exact from the code]

Uniform prior, target = character 0, every target flash gives LR = 11.
1. **Flash 1 (target's row group):** the 6 in-group characters rise from 1/72 = 0.0139 to
   0.0833 each; the 66 others fall to 0.00758.
2. **Flash 2 (target's column group):** row and column groups intersect only at the target.
   Unnormalized: target 11·11 = 121; 5 row-mates 11 each; 5 column-mates 11 each; 61 others 1.
   Total 292, so p(target) = 121/292 = **0.414**.

With the LM prior: if the LM gives the target 0.9945 (e.g. "สวัส"→ด), p_0 = 0.7·0.9945 + 0.3/72 =
**0.700**, 50× the uniform start of 0.0139. With the same scripted scores (+2 on target
flashes, −1 on the others, a = 1.2, b = −2.4), a uniform prior first reaches θ = 0.9 at
**flash 27**; the LM prior reaches it at **flash 3** (held to **6** by t_min). [SYNTHETIC
scores; illustrates the mechanism, not a measured speed-up]

### 5.8 The Transformer language model

**Eq. 10, input:** h⁰_i = E[x_i] + P[i], E ∈ ℝ^{1661×256}, P ∈ ℝ^{64×256}.

**Eq. 11, block (pre-norm residual), 6 times:**
u = h + MHA(LN(h)); h_next = u + W₂·ReLU(W₁·LN(u) + b₁) + b₂, W₁ ∈ ℝ^{256×1024}.

**Eq. 12, one attention head** (8 heads, head size 32, outputs concatenated then projected):

$$\text{head}(H) = \operatorname{softmax}\!\Big(\operatorname{mask}\big(Q K^\top \cdot d^{-1/2}\big)\Big)\, V, \quad Q = H W_Q,\ K = H W_K,\ V = H W_V$$

The causal mask sets future positions to −∞, so position i sees only characters ≤ i.
**Implementation note (Q&A magnet):** the scale uses d = 256 (model width), i.e. 1/16, not
the standard head size 32 (1/√32 ≈ 0.177) of Vaswani et al. Attention logits are therefore
2.83× smaller (softer attention). It comes from an early version of the tutorial the code
follows. The checkpoint was trained this way and must be served this way; the model trained
well (see loss). Changing it would require retraining. [VERIFIED: code]

**Eq. 13, output and loss:** P(x_{i+1} = v | x_≤i) = softmax(W_out·LN(h⁶_i) + b)_v;
training loss = mean −ln P(x_{i+1} | x_≤i) (cross-entropy in nats per character).

Conversions: val 1.163 nats/char = **1.68 bits/char**, perplexity e^1.163 = **3.20** (the
model is, on average, as uncertain as a uniform choice among 3.2 characters out of 1,661).
Train 1.535 nats, perplexity 4.64. These are full-vocabulary numbers on held-out Wikipedia,
not comparable to the matrix-restricted surprisal of the 16-item test.

**Parameter count** (verified to the checkpoint's 5,602,941): token embedding 425,216;
positional 16,384; 6 blocks × 788,992 = 4,733,952; final LayerNorm 512; output head 426,877.

Why val < train here (hypotheses, not proven): both were measured in eval mode on only 20
random batches, so estimates are noisy; the validation split is the last 10% of the shuffled
article stream and may be easier text; the run saw about 82M sampled positions, an
under-fitting regime far from memorization.

### 5.9 The n-gram language model

**Eq. 14** (`stopping/lm_prior.py`). For context h, let K = min(order − 1, |h|) = min(3, |h|).
For k = K, K−1, ..., 0, take the last k characters as the context u. At the first (longest) k
for which u was seen in the corpus:

$$P_k(c \mid u) = \frac{N(u, c) + \alpha}{\sum_{c' \in \mathcal{C}} \big(N(u, c') + \alpha\big)}, \qquad p(c \mid h) \propto \delta\,P_k(c \mid u) + (1-\delta)\frac{1}{N}, \quad \delta = \beta^{K-k}$$

α = 0.1, β = 0.4, order 4. This is inspired by stupid backoff but expressed as mixing with
uniform: the further it has to back off, the flatter the distribution. It is not Katz or
Kneser-Ney smoothing.

### 5.10 The classifier

**Eq. 15, xDAWN.** Model epochs as X = D·A + noise (D: stimulus design, A: the evoked
response). Choose spatial filters u maximizing the signal-to-signal-plus-noise ratio

$$\max_u \frac{u^\top \Sigma_{P}\, u}{u^\top \Sigma_X\, u}$$

(Σ_P: covariance of the averaged evoked response, Σ_X: covariance of the raw epochs), a
generalized eigenproblem; keep the top 4 filters per class.

**Eq. 16, augmented covariance.** For each epoch, stack the filtered class prototypes with the
filtered epoch and take its covariance: an SPD matrix of size **16 × 16** here (2 classes × 4
filters for the prototypes + 8 filtered epoch rows). [VERIFIED: computed]

**Eq. 17, tangent space.** With Σ_ref the Riemannian mean of the training covariances,

$$S_i = \log\!\big(\Sigma_{ref}^{-1/2}\, \Sigma_i\, \Sigma_{ref}^{-1/2}\big)$$

(matrix logarithm), then vectorize the upper triangle (off-diagonals weighted by √2): a vector
of length 16·17/2 = **136**. [VERIFIED: computed] The Riemannian mean minimizes the sum of
squared affine-invariant distances δ(A, B) = ‖log(A^{−1/2} B A^{−1/2})‖_F.

**Eq. 18, shrinkage LDA.** w = Σ_γ⁻¹(μ₁ − μ₀), Σ_γ = (1−γ)Σ̂ + γ·(tr Σ̂/p)·I with γ from the
Ledoit-Wolf formula; score s = wᵀx + w₀ (`decision_function`). The raw score is uncalibrated;
Eq. 6 calibrates it.

### 5.11 Evaluation math

**Eq. 19, Wolpaw ITR** (`stopping/replay.py`):

$$B = \log_2 N + P\log_2 P + (1-P)\log_2\frac{1-P}{N-1}\ \ \text{bits/selection}, \qquad \text{ITR} = B\cdot\frac{60}{T_{sel}}\ \text{bits/min}$$

Maximum log₂ 72 = 6.17 bits per selection. Standard assumptions: targets equiprobable, errors
spread evenly over the other N−1 characters, constant accuracy. With an LM prior, characters
are not equiprobable, so Wolpaw ITR is a conventional comparison figure, not the true
information rate.

Bits per selection at N = 72 [COMPUTED]: chance (1.4%) 0.00; 50% 2.10; 70% 3.44; 80% 4.22;
90% 5.09; 95% 5.58; 100% 6.17.

ITR under different time assumptions [COMPUTED, illustrative]:

| Scenario | s/selection | ITR at 90% | ITR at 80% |
|---|---|---|---|
| Fixed 5 repetitions, flash time only | 14.88 | 20.5 | 17.0 |
| Fixed 5 repetitions + 2 s cue + 3.5 s pause | 20.38 | 15.0 | 12.4 |
| Dynamic, ~20 flashes, flash time only | 2.48 | 123.0 | 102.0 |
| Dynamic, ~20 flashes + 2 s cue + 3.5 s pause | 7.98 | 38.2 | 31.7 |

Message: once flashes drop, fixed overheads (cue and pause) dominate. Always report which
time base an ITR uses. The project's tools report flash-time ITR by default (the same basis as
`evaluate_recording`) and wall-clock ITR with `--overhead-s`.

**Eq. 20, Wilson 95% interval** (`summarize_experiment.py`), z = 1.96:

$$\frac{p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Examples [COMPUTED]: 36/40 → 76.9%-96.0%; 9/10 → 59.6%-98.2%; 8/10 → 49.0%-94.3%. With 30-40
characters per session, expect intervals about ±10 percentage points.

**Eq. 21, expected calibration error** (10 equal-width bins):
ECE = Σ_b (n_b/n)·|acc_b − conf_b|.

**Other metrics:** AUC (probability a random target epoch outscores a random non-target
epoch) and balanced accuracy (mean of sensitivity and specificity), both robust to the 11:1
imbalance.

---

## 6. Evaluation methodology

### 6.1 Planned protocol [design; not yet executed]

1. **Calibration session** per subject: copy-spell a known sentence, no live inference.
2. **Fit** (§4.10): classifier, out-of-fold Platt calibration, manifest with the chosen LM.
3. **Held-out test session(s):** copy-spell 30-40 characters with live inference.
4. **Offline replay** (`stopping/evaluate_recording.py`) of the test recording under a grid of
   θ, t_min and LM types. The same recorded scores are replayed under every condition, so
   differences come from the stopping rule and the prior, not from session variability.
   This paired design is the right way to compare LMs and dynamic vs fixed stopping.
5. **Report** (`stopping/summarize_experiment.py --html`): accuracy with Wilson CI, mean and
   median flashes, flash-time and wall-clock ITR, stop-reason mix, LM effect.

### 6.2 Methodological choices to state explicitly

- **Held-out data.** Replay a session not used for fitting. The tool does not enforce it.
- **LM context in replay** uses the true preceding characters (ground-truth copy-spelling
  context). This isolates the stopping controller from error propagation. Live inference uses
  the committed text, so a wrong character can mislead the next prior. Say which is reported.
- **Seconds per flash** is measured from the recording's own flash markers, not assumed.
- **Rejected flashes** (quality gate, incomplete epochs) count toward neither t_min nor t_max.
- **Fixed-repetition baseline** can be expressed with the same controller by forcing the stop
  at exactly k repetitions (t_min = t_max = 24k). [OPEN: not yet run; confirm the CLI grid
  supports it]
- **Paired statistics** for within-subject comparisons, e.g. McNemar's test on per-character
  correctness and the Wilcoxon signed-rank test on flashes per character. [Suggested; not
  implemented]
- **Number of subjects:** [OPEN]

---

## 7. Results to date (status, not performance)

| Evidence | Tag | What it shows | What it does not show |
|---|---|---|---|
| BNCI2014-008/009 classifier comparison (§4.9) | [MEASURED-PUBLIC] | the classifier family separates target from non-target flashes on public P300 data (AUC 0.64-0.84) | anything about this headset, this paradigm, Thai, or character accuracy |
| LM comparison, 16 items (§4.12) | [MEASURED-LM] | the full-corpus Transformer predicts common Thai continuations better than the n-gram (81% vs 75% top-1, 0.63 vs 0.96 bits) | effect on spelling speed; significance |
| Transformer training (§4.12) | [MEASURED] | trains stably; val 1.163 nats/char, no sign of overfitting | downstream benefit |
| LM latency (§4.12) | [MEASURED-SW, Mac] | 6.7 ms per prior vs a 2,000 ms budget | Windows latency [OPEN] |
| Relay protocol decoder | [VERIFIED] | reproduces g.tec's worked example exactly | real-headset reliability |
| First real headset connection (§4.5) | [MEASURED-HW] | 2,298 packets, 0 lost; panel "EEG connected" | EEG quality (headset not worn) |
| Longer bench run (§4.5) | [MEASURED-HW] | 5.0% sample loss in two bursts | cause (interference suspected) |
| Later connection attempts (§4.5) | [MEASURED-HW] | link not established | [OPEN] |
| Demo sessions ("สวัสดี" 6/6; "สวัสดีครับ" 9/10; ITR 148-171 bits/min flash-time) | **[SYNTHETIC]** | the full pipeline, real Transformer and report generator run end to end; characters the LM ranked 1st took fewer flashes (e.g. 8.7 vs 16.9) | **anything about real performance: scores were scripted** |

**Does not exist yet:** any real-subject calibration, any real-subject online spelling
session, any real accuracy, flashes-per-character or ITR figure, any comparison of the three
LMs on real data, any fixed-vs-dynamic comparison on real data.

---

## 8. Limitations, risks and open items

1. **No real-subject data yet.** Every claim about benefit is a hypothesis.
2. **Open-loop stimulus.** The stimulus always runs 5 repetitions. Early stops are computed but
   the subject still waits, so ITR gains are potential, not realized, until the stimulus
   accepts a stop signal. Flash-time ITR from replay describes the closed-loop system.
3. **Assumption A1 is violated in practice.** At a 124 ms SOA, ERPs from successive flashes
   overlap; attention, fatigue and slow drifts correlate scores across flashes. The posterior
   will tend to be overconfident. Mitigations in place: LLR clipping, t_min, out-of-fold Platt
   calibration, and a 7-flash minimum gap between repeats of a character. Check on real data
   whether threshold-stopped characters are correct at least θ of the time.
4. **Calibration drift.** Platt parameters come from one calibration session; electrode
   impedance, fatigue and attention change across sessions.
5. **LM domain and context.** Trained on Wikipedia, not conversational or spelled text.
   Live, committed errors become context and can mislead the next prior. λ is fixed at 0.7,
   and the LM alone can raise a character to at most 0.704.
6. **Matrix coverage.** No space and several rarer characters (§4.2); the character list is
   not yet verified against the reference source.
7. **Hardware and timing.** Mac Bluetooth reliability unresolved; the live stream's channel
   labels not yet checked (H2/H13); the PsychoPy SOA on the Mac is 133 ms, not 124 ms;
   flip-to-light lag unmeasured (H7). Windows machine timing and latency not yet measured.
8. **Small samples.** 30-40 characters give ±10-point confidence intervals.
9. **Wolpaw ITR assumptions** do not hold with a language model (non-equiprobable targets).
10. **Unreconciled BNCI numbers** (0.703 vs 0.769 AUC for FgMDM on 008).
11. **Implementation details an examiner may probe:** attention scaling by √256 instead of √32
    (§5.8); Platt scaling uses scikit-learn's default logistic regression, which includes an
    L2 penalty (C = 1.0) and no Platt target smoothing (small effect with one feature and
    thousands of samples).
12. **Ethics approval** for human data collection is not recorded in the repository. [OPEN]

---

## 9. Suggested presentation outline (for the agent building slides)

Each slide: key message, content, suggested visual, source section, caveat to keep.

| # | Slide | Key message | Visual | Source |
|---|---|---|---|---|
| 1 | Title | Thai P300 speller with LM-informed Bayesian dynamic stopping | headset photo or matrix | §1 |
| 2 | Motivation | typing by attention; spellers are slow because they repeat blindly | timeline of a fixed 20 s selection | §2.1, §2.3, §5.11 |
| 3 | The P300 | an ERP ~300 ms after an attended rare flash | textbook ERP sketch (label as schematic) | §2.1 |
| 4 | Why Thai | combining marks cost a selection each but are predictable | "สวัสดี" as 6 selections | §2.2 |
| 5 | The matrix | 72 characters, 8×9, deliberate substitutions | the grid | §4.2 |
| 6 | Checkerboard paradigm | 24 flashes/repetition, each character twice, neighbours never together, repeats ≥ 7 flashes apart | colored grid + virtual 6×6 matrices | §4.3 |
| 7 | System architecture | headset → LSL → recorder / inference / panel | §3.1 diagram | §3 |
| 8 | Signal path | epoch 0-0.8 s, 1-24 Hz, quality gate, xDAWN → tangent space → LDA | pipeline diagram | §4.7-4.9 |
| 9 | Classifier validation | works on public P300 data; flash-level only | §4.9 table | §4.9 (keep caveats) |
| 10 | Core idea | inference over 72 hypotheses, one flash at a time | posterior bar chart evolving | §5.2 |
| 11 | Prior | LM → marginalize (Eq. 1) → blend with uniform (Eq. 2); floor and ceiling | equation + two bar charts | §5.3 |
| 12 | Update | exact Bayes under A1, group-only log update (Eq. 3-5) | 6 bars rising, 66 falling | §5.4 |
| 13 | Score → evidence | Platt + base-rate correction (Eq. 6-8) | the §5.5 intuition table | §5.5 |
| 14 | Stopping | θ, t_min, t_max (Eq. 9) | posterior max crossing θ | §5.6 |
| 15 | Worked example | 0.0139 → 0.083 → 0.414; LM start 0.700 | step chart | §5.7 |
| 16 | The language model | 5.6M-parameter character GPT on Thai Wikipedia, marginalized to 72 | architecture sketch + loss curve | §4.12, §5.8 |
| 17 | LM comparison | Transformer beats n-gram on a 16-item test (suggestive) | §4.12 table | §4.12 (state n = 16) |
| 18 | Why it should be faster | ITR scenarios; overheads matter | §5.11 table | §5.11 (label computed) |
| 19 | Engineering for trust | train/serve parity, hash-verified manifest, strict loader, 696 tests, full experiment logs | icons/checklist | §4.13-4.17 |
| 20 | Status | pipeline complete; hardware bring-up underway; no real-subject results yet | §7 table | §7 |
| 21 | Evaluation plan | paired replay across θ, t_min, LM; held-out sessions | protocol flow | §6 |
| 22 | Limitations and next steps | §8 top items; closed-loop stimulus; data collection | list | §8 |
| 23 | Candidate contributions | the three framings (owner to choose) | table | §2.5 |

Backup slides: derivation of Eq. 7; checkerboard spacing proof; tangent-space math; Transformer
parameter breakdown; Mac relay protocol and clock; quality-gate thresholds; the synthetic demo
report (clearly labelled synthetic); the full Q&A in Appendix A.

---

# Appendices

## Appendix A. Q&A preparation bank

Answers are written to be said aloud. Section references point to the supporting detail.

### A.1 Motivation and novelty

**Q1. What is new compared with Pananookooln et al. 2022?**
The checkerboard paradigm, the matrix and the idea of a Thai P300 speller come from that work
and are reimplemented, not claimed. What is new is the decision layer: a Bayesian posterior
over all 72 characters updated after every flash, stopping as soon as it is confident, and
starting each character from a Thai character-level Transformer language model. The pipeline
is also fully open and reproducible. The exact contribution framing is still being decided
(§2.5). [Their stopping protocol and reported results: verify from the paper before comparing.]

**Q2. Dynamic stopping and language models in P300 spellers are not new. Why is this?**
Correct, both ideas have prior work in English spellers (Appendix G, verify before citing).
What is specific here is the Thai setting, where combining marks each cost a full selection
but are highly predictable; a character-level Transformer conditioned on the speller's
alphabet by marginalization (Eq. 1); and the exact group-structured update for the
checkerboard paradigm. Do not claim "first" without a literature check.

**Q3. Why not add word-completion cells to the matrix?**
It was explored and rejected on its merits: the checkerboard needs an even number of cells
and an odd column count, word and character probabilities are different quantities that
cannot share one normalized prior, it needs a matrix rebuilt every trial, word generation is
a harder separate model, and the character prior already delivers the same mechanism without
those risks. It belongs in future work (Appendix F.3).

**Q4. Why Thai?**
Thai's combining marks, vowel signs and tone marks make many characters close to determined by
context (after "สวัส" the Transformer puts 99.4% on "ด"). That is exactly the situation where
an LM prior should save the most flashes. Thai is also less studied than English. [Literature
coverage claim: verify.]

**Q5. Why a P300 speller rather than SSVEP or motor imagery?**
It works with the 8-channel consumer headset available, needs no user training, supports 72
choices, and is a well-studied paradigm with a Thai baseline to build on. It was the advisor's
first project in a planned sequence (§2.6).

**Q6. Could this help people with ALS?**
Possibly in the long run, but no clinical claim is made. The project starts with healthy
subjects. One public validation dataset (BNCI2014-008) is from ALS patients, which is why its
AUC is lower.

### A.2 Paradigm and stimulus

**Q7. Why the checkerboard paradigm instead of rows and columns?**
Adjacent characters never flash together, which reduces adjacency errors, and the same
character never flashes twice within 7 flashes (868 ms), longer than the 800 ms epoch, which
avoids overlapping target responses (§4.3). Both properties were verified on the code.

**Q8. Why are there 6 characters per flash?**
The 36 characters of each color are arranged in a 6×6 virtual matrix and its rows and columns
are flashed. So each flash covers 6 of 72, π = 1/12, and each character is flashed twice per
24-flash repetition.

**Q9. What is the flash timing?**
Nominal 62 ms on, 62 ms off (124 ms SOA), 5 repetitions (120 flashes), a 2 s cue and a 3.5 s
pause. On the Mac's PsychoPy window the measured SOA is 133 ms because of frame rounding, so
offline analysis measures seconds per flash from the markers instead of assuming 124 ms.

**Q10. Why 8 rows by 9 columns and not 9 by 8?**
With an odd column count, flat-index parity equals true checkerboard parity. With 9×8 the
coloring degenerates to column stripes and neighbours can co-flash (54 violations measured).

**Q11. How do you know the intended character?**
Every session is copy-spelling: each character is cued for 2 s before its flashes and the cue
is logged in the `trial_start` marker. The decoder never reads it; it is used only to score.

**Q12. Can the user correct mistakes or type spaces?**
Not currently: the matrix has no space or delete cell (§4.2). Copy-spelling sessions do not
need them. Free spelling with correction is future work.

### A.3 Signal processing

**Q13. Why a 0-0.8 s epoch?**
It covers the P300 search window (0.25-0.55 s) with margin for later components, gives 200
samples at 250 Hz, and is identical in training and live use.

**Q14. Why 1-24 Hz?**
It is the band of MOABB's P300 paradigm, with which the classifier pipelines were validated on
the public datasets.

**Q15. Zero-phase filtering uses future samples. Is that valid online?**
Yes, at the epoch level: the service waits until the whole padded window (flash + 0.8 s + 0.5 s)
has arrived, then filters only that window. The cost is a decision delay of about 1.3 s per
flash. Training and serving use the same code, so there is no mismatch.

**Q16. Why no baseline correction online?**
The 1 Hz high-pass already removes slow offsets, and the same epoch definition is used in
training and serving. The public-data EDA path used a −0.2-0 s baseline to fix a small Fz
artifact; that is a separate analysis path.

**Q17. Why throw away a whole flash when one channel is bad?**
xDAWN combines all 8 channels, so one railed electrode contaminates every component and would
produce a confident wrong score. The classifier cannot drop a channel without refitting. A
rejected flash is simply not counted, so the controller waits for more evidence.

**Q18. How do you handle eye blinks?**
There is no explicit artifact removal. Blinks usually show as "noisy" (yellow) and are kept;
railed or flat channels are rejected. The spatial filters and calibration absorb some artifact
variance. Blink handling on real data is a limitation to evaluate.

**Q19. Where do the 1 µV / 200 µV / 400 µV thresholds come from?**
From the recorder's plausibility check, not tuned on Unicorn data. They are shared by the
panel and the gate so the operator and the decoder never disagree about a channel.

### A.4 Classifier

**Q20. Why xDAWN + tangent space + LDA instead of a deep network?**
Per-subject calibration data is small (a few thousand epochs at most); this pipeline is fast,
well established for P300, and was validated on public data. The classifier is deliberately
not the contribution.

**Q21. FgMDM had better balanced accuracy on BNCI2014-008. Why not use it?**
Both have identical AUC by construction. Balanced accuracy depends on the hard decision
threshold, which this system never uses: the continuous score is calibrated and fed to the
posterior. For that purpose AUC is the relevant metric.

**Q22. Why is AUC only about 0.70 on BNCI2014-008?**
ALS patients, single-trial flash-level classification, and cross-subject folds (GroupKFold by
subject trains on other people). The live system calibrates on the same subject, which is
easier, so these public numbers are conservative. There is also an unreconciled earlier run
(0.769), so treat them as indicative.

**Q23. How big are the features?**
Each epoch becomes a 16×16 covariance matrix (8 prototype rows + 8 filtered epoch rows), and the
tangent-space vector has 136 entries (§5.10).

**Q24. How is class imbalance handled?**
Metrics are AUC and balanced accuracy; Platt scaling learns the base rate, and the
log-likelihood ratio removes it again (Eq. 7), so imbalance does not bias the posterior.

**Q25. Why group cross-validation by trial when fitting?**
Epochs from the same trial share state and noise. Splitting them across folds would leak and
inflate the out-of-fold scores that Platt scaling is fitted on.

### A.5 Calibration

**Q26. Why Platt scaling and not isotonic regression?**
Two parameters are stable with limited calibration data, and the mapping is smooth and
monotone, so log-likelihood ratios change continuously with the score. Isotonic regression
needs more data and produces steps.

**Q27. Why fit it on out-of-fold scores?**
A model scores its own training data too confidently. Calibrating on in-sample scores would
inflate every likelihood ratio and cause early wrong stops.

**Q28. What if calibration is poor?**
ECE is computed and logged at fit time. Clipping (±8) and t_min limit the damage, and the
experiment logs let you check whether threshold stops are correct at least θ of the time.

**Q29. Is your Platt scaling the textbook version?**
Almost. It uses scikit-learn's default logistic regression, which includes a small L2 penalty
and no target smoothing. With one feature and thousands of samples the effect is small.

### A.6 The stopping math

**Q30. Derive the log-likelihood ratio.**
Platt gives logit P(target | s) = a s + b. Bayes at the flash level says that logit equals the
log-likelihood ratio plus the log prior odds of a target flash, ln[π/(1−π)]. So the LLR is
a s + b + ln[(1−π)/π] (Eq. 7). Here π = 1/12, so the correction is ln 11 ≈ 2.40.

**Q31. Why add the evidence only to the flashed characters?**
If the target is in the flashed group, the score comes from the target distribution; otherwise
from the non-target one. Dividing every hypothesis by the non-target likelihood, which is
common to all, leaves the in-group characters multiplied by the likelihood ratio and the rest
unchanged. That is exact Bayes under conditional independence (Eq. 3-4).

**Q32. Is conditional independence realistic?**
No. Responses overlap at a 124 ms SOA, and attention and fatigue correlate scores. The
posterior will be somewhat overconfident. Mitigations: clipping, t_min, out-of-fold
calibration, the 7-flash repeat spacing. Whether it matters is an empirical question the logs
can answer.

**Q33. Why blend the LM with a uniform floor?**
So an LM mistake can never make the true character impossible: it keeps at least 0.3/72 of
probability. And so the LM alone can never stop a trial: it can raise a character to at most
0.704, below θ = 0.9.

**Q34. How were λ = 0.7, θ = 0.9 and t_min = 6 chosen?**
They are defaults, not tuned. The replay tool sweeps θ and t_min (and λ) on recorded data, and
tuning them on real sessions is planned.

**Q35. Why clip the LLR at ±8?**
It caps any single flash at changing a group's odds by about 3,000×, which bounds the effect of
artifacts and outlier scores.

**Q36. What happens if the threshold is never reached?**
With default settings the stimulus runs 120 flashes, fewer than t_max = 170, so the trial ends
and the service commits the most probable character with the reason `trial_ended`. Fallback
commits are logged and shown differently from confident ones.

**Q37. Is this an optimal stopping rule?**
It is a fixed-threshold posterior rule, a multi-hypothesis relative of Wald's sequential
probability ratio test. No optimality is claimed. An adaptive, boundary-aware threshold was
considered and descoped.

**Q38. Why work in log space?**
Products of many small probabilities underflow; sums of log-probabilities with log-sum-exp
normalization are stable even for extreme evidence.

**Q39. Walk me through a number.**
Uniform start 0.0139. One flash with likelihood ratio 11 containing the target: the 6 flashed
characters go to 0.083. A second flash (its column) intersects only at the target: 0.414. With
an LM giving 0.99, the start is already 0.70 (§5.7).

### A.7 The language model

**Q40. Why character-level and not word-level?**
Written Thai has no spaces between words, and the matrix is made of characters, including
marks. A character model gives exactly the next-cell probability that the controller needs.

**Q41. Why train on the full vocabulary and then restrict to 72?**
So the model learns real Thai, including characters outside the matrix as context. Restricting
by renormalizing is exact conditioning on "the next character is typeable" (Eq. 1).

**Q42. Why a block size of 64?**
Sessions are about 40 characters at most, so a 64-character window covers the whole session
(train/serve parity) and keeps attention cheap.

**Q43. Why not a large pretrained model?**
Large pretrained models are slower and usually tokenize into subwords, which do not map cleanly
onto single Thai code points such as tone marks. A small character model predicts the next code
point directly, trains in 26 minutes and runs in 7 ms.

**Q44. Is the 16-item test in the training data?**
The words are common and certainly appear in Wikipedia. The test checks whether the model
learned common continuations, not generalization; held-out generalization is the validation
loss (1.163 nats/char, perplexity 3.2). The real test is spelling speed on subject data.

**Q45. Why is validation loss lower than training loss?**
Both were estimated in eval mode on 20 random batches, so they are noisy; the validation text
may simply be easier; and the model saw about 82M positions, far from memorizing. It is not a
sign of a bug.

**Q46. What about archaic characters like ฃ and ฅ?**
They are force-included in the vocabulary. Otherwise they would collapse into `<unk>` and the
prior for those two cells would be meaningless.

**Q47. Your attention scaling looks non-standard.**
Yes: it divides by √256 (model width) instead of √32 (head size), inherited from an early
version of the tutorial. It makes attention softer; the model trained well; fixing it means
retraining (§5.8).

**Q48. What if the matrix changes?**
The checkpoint only knows the matrix it was trained with. Loading refuses a checkpoint missing
any matrix character, and the fix is retraining on Colab.

**Q49. Why is the prior after "ขอบ" spread out?**
Several common words continue from it (ขอบคุณ, ขอบใจ, ขอบเขต). A calibrated LM should be
uncertain there; forcing confidence would cost flashes when it guesses wrong.

**Q50. What happens after a decoding error?**
The wrong character becomes context and can mislead the next prior. The uniform floor keeps
the right character reachable. Replay uses the true preceding text to isolate this effect;
live logs record what actually happened.

### A.8 Evaluation

**Q51. Is your ITR flash time only?**
By default yes, the same basis as the replay tool; `--overhead-s` gives wall-clock ITR. Both
should be reported, because once flashes drop the 2 s cue and 3.5 s pause dominate (§5.11).

**Q52. How many characters and subjects do you need?**
30-40 characters per session give confidence intervals of about ±10 points. Comparisons are
paired (the same recorded flashes replayed under each condition), which needs fewer subjects
than independent groups. The number of subjects is still open.

**Q53. Why the Wilson interval?**
It has good coverage for small samples and accuracies near 0 or 100%, where the normal
approximation fails.

**Q54. How will you compare the language models fairly?**
Replay the same recorded scores under uniform, n-gram and Transformer priors. Only the prior
changes, so differences in flashes and accuracy are attributable to it.

**Q55. What is chance?**
1/72 = 1.4% accuracy, which is 0 bits per selection.

**Q56. Does Wolpaw ITR make sense with a language model?**
Only as a convention: it assumes equiprobable targets, which an LM explicitly breaks. Report it
for comparability, alongside correct characters per minute.

### A.9 Engineering and hardware

**Q57. Why LSL?**
It is the standard for time-synchronized multi-stream research recording, and it works across
processes and operating systems. All streams share one clock.

**Q58. How do you guarantee every process uses the same model?**
The session manifest lists every artifact with its sha256, re-verified at load, and its
fingerprint is shared between processes; any mismatch stops the session.

**Q59. Why can the inference service refuse to start?**
A classifier served epochs cut differently from its training data scores worse silently. So the
service refuses (exit 2) unless the channels, sample rate and epoch window match exactly.

**Q60. Why write your own Mac relay?**
g.tec's software is Windows-only. The relay speaks g.tec's published Bluetooth protocol,
reproduces their worked example exactly, and stamps samples from the device's counter. It has
not passed the hardware checks yet (§4.5).

**Q61. How do you know the timestamps are right?**
LSL clock synchronization, counter-based stamping on the Mac path, integrity checks for gaps
and non-monotonic stamps, and dejitter deliberately off because it was measured hiding problems.
Absolute screen-to-marker lag needs a photodiode (H7).

**Q62. Is the Unicorn good enough for P300?**
Its channels include the classic P300 speller sites (Fz, Cz, Pz, PO7, Oz, PO8). Published
validation of this headset exists (Appendix G, verify). It is consumer grade, so noise and
contact quality are real risks.

**Q63. The live EEG plot looked strange. Why?**
The display normalizes each channel to fill its row, so it shows shape, not size, and the
headset was not being worn, so the electrodes picked up ambient electrical noise. The scalp map
colours use real microvolt thresholds and are the reliable indicator. The hardware amplifier is
fixed (±750 mV, 24-bit, 0.0894 µV per count), and the software applies no other gain.

**Q64. What about the 5% packet loss?**
It came in two short bursts in an early bench test, consistent with Wi-Fi interference. Lost
packets show up as timestamp gaps, and the affected epochs are rejected rather than corrupted.
The 30-minute gap test decides whether the Mac link is usable.

**Q65. How is the software tested?**
696 automated tests, including fault injection across processes, real-LSL tests on randomized
stream types, and relay tests against a protocol-accurate fake headset. Hardware checks
H1-H14 cover the rest.

### A.10 Ethics, limits and future work

**Q66. Do you have ethics approval?**
[OPEN: not recorded in the repository. Answer from the actual status; do not guess.]

**Q67. What are the main limitations?**
No subject data yet; open-loop stimulus; conditional independence violated; untuned λ, θ,
t_min; small samples; hardware bring-up unfinished (§8).

**Q68. What would you do next?**
Collect calibration and test sessions; make the stimulus stop when the controller stops;
tune the parameters by replay; check posterior calibration; compare the three LMs; then
consider adaptive thresholds and information-based flash ordering (stubs exist, out of scope).

---

## Appendix B. Parameter reference

| Parameter | Value | Where |
|---|---|---|
| Characters | 72, 8×9, row-major | `collection/characters_th.json` |
| Channels | Fz C3 Cz C4 Pz PO7 Oz PO8 | `pipeline/constants.py` |
| Sampling rate | 250 Hz | `pipeline/constants.py` |
| Flash / ISI / SOA | 62 / 62 / 124 ms | stimulus defaults |
| Repetitions | 5 (120 flashes) | stimulus default |
| Cue / inter-selection | 2000 / 3500 ms | stimulus defaults |
| Flashes per repetition | 24 (6 characters each) | `collection/flash_sequence.py` |
| Epoch | 0-0.8 s (200 samples) | `pipeline/epoching.py` |
| Band-pass | 1-24 Hz, Butterworth order 4, zero phase | `pipeline/epoching.py` |
| Epoch padding | 0.5 s each side | `pipeline/epoching.py` |
| Gap tolerance | 1.5 × sample period | `pipeline/epoching.py` |
| Quality thresholds | flat < 1 µV, noisy > 200 µV, railed > 400 µV, flat 100 ms segment | `pipeline/quality.py` |
| Gate policy | skip | manifest default |
| xDAWN filters | 4 per class | `classifiers/xdawn_cov_ts_regLDA.py` |
| LDA | lsqr solver, Ledoit-Wolf shrinkage | same |
| Fit minimums | 3 trials, 10 target, 10 non-target epochs | `pipeline/fit_from_recording.py` |
| CV folds | 5, grouped by trial | same |
| ECE bins | 10 | `stopping/calibrate.py` |
| λ | 0.7 | `stopping/bayesian_stopping.py`, manifest |
| θ | 0.90 | same |
| t_min / t_max | 6 / 170 | same |
| LLR clip | ±8 | same |
| π | \|group\| / 72 = 1/12 | inference service, replay |
| n-gram | order 4, α 0.1, β 0.4 | `stopping/lm_prior.py` |
| Transformer | 64 context, 256 dim, 8 heads, 6 layers, dropout 0.1, 1,661 vocab | checkpoint config |
| Marginalization ε | 1e-8 | `stopping/transformer_lm.py` |
| Prior validation tolerance | sum within 1e-6 | `pipeline/language_model.py` |
| Stale epoch | 5 s | `pipeline/inference_service.py` |
| Ready heartbeat | 2 s | same |
| Panel waits | service 20 s, recorder 35 s | `collection/gui_logic.py` |
| Relay | warm-up 2 s, window 15 s, slew 200 ppm, silence 3 s (5 s before first packet) | `collection/unicorn_relay.py` |
| Wilson z | 1.959964 | `stopping/summarize_experiment.py` |

---

## Appendix C. Glossary

| Term | Meaning |
|---|---|
| AUC | Area under the ROC curve: probability a random target epoch scores above a random non-target |
| Balanced accuracy | mean of sensitivity and specificity |
| Checkerboard paradigm | flashing rows and columns of two virtual matrices built from alternately coloured cells |
| Copy spelling | the subject spells a known cued text, so accuracy can be scored |
| ECE | expected calibration error: average gap between predicted confidence and observed accuracy |
| Epoch | the EEG segment after one flash (here 0-0.8 s) |
| ERP | event-related potential: brain voltage time-locked to a stimulus |
| ISI / SOA | inter-stimulus interval / stimulus onset asynchrony (flash-to-flash time) |
| ITR | information transfer rate (Wolpaw), bits per minute |
| LDA | linear discriminant analysis |
| LLR | log-likelihood ratio: log of how much more likely a score is under "target" than "non-target" |
| LSL | Lab Streaming Layer: time-synchronized data streaming |
| Marginalization (here) | restricting the LM's distribution to the 72 matrix characters and renormalizing |
| Oddball paradigm | rare relevant stimuli among frequent irrelevant ones |
| P300 | positive ERP about 250-550 ms after an attended rare stimulus |
| Platt scaling | logistic regression from a raw score to a calibrated probability |
| Posterior | probability of each character given all evidence so far |
| Riemannian mean / tangent space | geometry for averaging and flattening covariance matrices |
| SPD | symmetric positive definite (matrix) |
| Train/serve parity | identical preprocessing in training and in live use |
| xDAWN | spatial filtering that enhances the evoked response |

---

## Appendix D. Repository map

```
start_speller.sh, setup_windows.sh, SETUP.txt, WINDOWS_SETUP.md   launch and setup
collection/
  session_gui_qt.py, gui_logic.py     operator panel; logic incl. SessionLogger
  checkerboard_stimulus.py            PsychoPy stimulus (precise flip timestamps)
  board_qt.py, board_schedule.py      in-panel stimulus board and its schedule
  flash_sequence.py                   checkerboard grouping (pure)
  record_session.py                   recorder (CSV + log + integrity)
  mock_headset.py                     synthetic EEG with P300s (demo mode)
  unicorn_relay.py, unicorn_protocol.py   Mac headset relay and protocol logic
  characters_th.json                  the 72-character matrix
pipeline/
  constants.py quality.py events.py manifest.py epoching.py experiment_log.py   shared contracts
  inference_service.py                live decoding
  fit_from_recording.py               calibration -> classifier + Platt + manifest + fit report
  classifier_io.py, language_model.py checkpoint I/O; LM factory
stopping/
  bayesian_stopping.py                the controller (owner-implemented)
  calibrate.py lm_prior.py transformer_lm.py replay.py
  evaluate_recording.py               real-recording replay and sweeps
  summarize_experiment.py             logs -> summary / HTML report
  gpt_dev.ipynb                       Transformer training (Colab only)
  compare_transformers.py model_comparison.py model_comparison_results.json
classifiers/                          the five validated pipelines, benchmark_all.py
config.py data_loader.py              public-dataset (MOABB) path, in volts
tools/                                fake inference publisher, fake Unicorn headset, e2e smoke test
tests/                                696 tests
docs/INTERFACES.md SYSTEM_OVERVIEW.md TESTING.md CLAUDE.md   references
```

---

## Appendix E. Data formats

**Marker JSON** (one per LSL sample):
```json
{"event": "trial_start", "trial_id": 3, "target_char_idx": 19, "target_char": "ด"}
{"event": "flash", "trial_id": 3, "flash_idx": 0, "group": [22, 10, 54, 2, 44, 24], "is_target": false}
{"event": "trial_end", "trial_id": 3}
```

**Inference events** (strictly validated): `posterior_update` {trial_id, flash_idx, max_prob,
leading_char_idx, theta_current}; `character_committed` {trial_id, char_idx, char,
n_flashes_used, reason}; `flash_rejected` {trial_id, flash_idx, reason, channels};
`service_status` {state, detail, manifest_fingerprint}.

**Session manifest** (abridged):
```json
{"schema_version": 1,
 "characters":  {"path": "collection/characters_th.json", "sha256": "<64 hex>"},
 "classifier":  {"path": "models/subject_001_classifier.joblib", "sha256": "<64 hex>"},
 "calibration": {"path": "models/subject_001_platt.json", "sha256": "<64 hex>"},
 "language_model": {"type": "transformer", "checkpoint": {"path": "models/thai_gpt_checkpoint.pt", "sha256": "<64 hex>"}},
 "stopping": {"lam": 0.7, "theta": 0.9, "t_min": 6, "t_max": 170, "llr_clip": 8.0},
 "quality_gate": {"policy": "skip"}}
```

**Inference log record at the start of a character** (abridged, from a synthetic run):
```json
{"event": "trial_start", "trial_id": 5, "context": "สวัส",
 "lm_prior": ["...72 floats..."], "initial_posterior": ["...72 floats..."],
 "lm_prior_top5": [["ด", 0.994491], ["ต", 0.00134782], ["ว", 0.000742047], ["ส", 0.000595471], ["บ", 0.000526763]],
 "target_char_idx": 19, "target_char": "ด", "target_prior_prob": 0.994491, "target_prior_rank": 1}
```

---

## Appendix F. Design decision log

### F.1 Timeline (from the commit history)

| Date | Decision or milestone |
|---|---|
| 2026-08-16 | repository started |
| 2026-09-11 | baseline collection, stopping module, classifiers; hardened acquisition; inference service and panel; in-window board |
| 2026-09-15 | **montage corrected** to the Unicorn's real cap (4 of 8 sites changed); earlier data invalidated |
| 2026-09-17 | **stopping controller implemented by hand**; real-recording ITR evaluation |
| 2026-09-18 | Thai stimulus font bundled and verified |
| 2026-09-20 | Windows deployment script |
| 2026-09-21 | digits 6/7 replaced by ั and ็; word-completion cells rejected |
| 2026-09-24 | **Transformer LM prior**, experiment logging, Mac relay; Windows de-risking |
| after | HTML experiment reports (charts, findings, benchmarks) [uncommitted at time of writing] |

### F.2 Decisions with reasons

- **Strict checkpoint loader.** Silent train/serve mismatch degrades accuracy without any error.
- **Out-of-fold calibration.** In-sample scores inflate every likelihood ratio.
- **Quality gate "skip".** One bad electrode contaminates all xDAWN components.
- **Dejitter off.** Measured to hide out-of-order timestamps by up to 18.7 ms.
- **Group-only log update.** Exact Bayes under the independence assumption; touches 6 entries.
- **Uniform floor (λ = 0.7).** LM errors cost flashes, never make a character impossible; LM
  alone cannot stop a trial.
- **Character-level Transformer, marginalized.** Learns real Thai; exact conditioning to the
  alphabet.
- **Block size 64.** Matches session length; train/serve parity.
- **Colab-only training.** Owner's rule; the laptop is not used for training.
- **Counter-based timestamps on the Mac.** Bluetooth bursts would smear ERPs.
- **Lazy torch import.** Uniform and n-gram manifests never need torch.

### F.3 Rejected: predictive whole-word cells

Replacing some cells with LM-predicted word completions was rejected because: the checkerboard
requires an even cell count and an odd column count; word and next-character probabilities
are different quantities that cannot share one normalized prior without reformulation; it
needs a per-trial rebuilt matrix while everything assumes fixed, hash-verified content; word
generation is a separate, harder model; and the character prior already delivers the benefit.
Discussion-section future work only.

### F.4 Descoped

Boundary-aware (adaptive) thresholds; information-gain flash ordering (`stopping/flash_selector.py`
stubs); any use of g.tec's Unicorn Speller (never, as a matter of principle).

---

## Appendix G. Related work to cite (ALL UNVERIFIED: check each against the source)

The project has a documented history of fabricated AI citations. Treat every entry as a lead,
not a reference, until checked.

| Topic | Likely source (verify author, year, venue, claims) |
|---|---|
| Original P300 speller | Farwell & Donchin, 1988 |
| Checkerboard paradigm | Townsend et al., 2010 |
| Thai P300 speller (the baseline) | Pananookooln et al., 2022 (AIT Brain Lab) |
| xDAWN spatial filtering | Rivet et al., 2009 |
| Riemannian geometry / tangent space for BCI | Barachant et al., 2012-2013 |
| MOABB benchmark framework | Jayaram & Barachant, 2018 |
| BNCI2014-008 (ALS P300 data) | Riccio et al., 2013 |
| BNCI2014-009 | Aricò et al., 2014 |
| Shrinkage covariance | Ledoit & Wolf, 2004 |
| Platt scaling | Platt, 1999 |
| ITR definition | Wolpaw et al., 1998 / 2002 |
| Dynamic stopping in P300 spellers | e.g. Schreuder et al., 2013; Throckmorton et al., 2013; Mainsah et al., 2015 |
| Language models in P300 spellers | e.g. Speier et al., 2012; Kindermans et al., 2012 |
| Transformer | Vaswani et al., 2017 |
| Character-level GPT tutorial | Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video, 2023) |
| Unicorn Hybrid Black validation | Pontifex et al., 2023, Psychophysiology |
| Thai Wikipedia dataset | `pythainlp/thai-wiki-dataset-v3` (HuggingFace), CC-BY-SA-3.0 |

---

## Appendix H. Hardware-in-the-loop checks (`TESTING.md`)

| Check | What it verifies | Status |
|---|---|---|
| H1 | stream discovery and shape | [OPEN] |
| H2 | live channel labels match the montage | [OPEN] |
| H3 | units: 1-200 µV peak-to-peak on a worn headset | [OPEN] |
| H4 | timestamp behaviour; decide on dejitter | [OPEN] |
| H5 | real dropout handling | [OPEN] |
| H6 | electrode lift turns red within ~200 ms; gate rejects | [OPEN] |
| H7 | photodiode: marker-to-screen lag and jitter | [OPEN] |
| H8 | Thai rendering in the stimulus | verified on Mac; Windows [OPEN] |
| H9 | no trials without `trial_start` | [OPEN] |
| H10 | live inference end to end | [OPEN] |
| H11 | 10-minute sustained load; lag stays ~1.3 s | [OPEN] |
| H12 | Mac relay 30-minute sample-counter gap test | [OPEN] (early runs: 0% and 5% loss) |
| H13 | Mac relay montage and units on the live stream | [OPEN] |
| H14 | Mac relay end-to-end timing | [OPEN] |

---

## Appendix I. Reproduction commands

```bash
PY="$(conda info --base)/envs/p300speller/bin/python"
./start_speller.sh --demo                                 # full pipeline on synthetic EEG
QT_QPA_PLATFORM=offscreen $PY -m pytest tests/ -q          # 696 passed, 3 skipped
$PY -m stopping.model_comparison                          # LM comparison table (§4.12)
$PY -m pipeline.fit_from_recording --recording collection/recordings/<file>.csv --id subject_001 \
    --lm-transformer-checkpoint models/thai_gpt_checkpoint.pt
$PY -m stopping.evaluate_recording --manifest session_manifest.json \
    --recording collection/recordings/<held_out>.csv --theta 0.7 0.8 0.9 0.95 --t-min 1 3 6
$PY -m stopping.summarize_experiment --recording collection/recordings/<session>.csv --html report.html
$PY collection/unicorn_relay.py --duration 1800 --report logs/gap_test.json    # Mac, H12
```

Transformer training: open `stopping/gpt_dev.ipynb` on Google Colab with a GPU runtime (never
locally).
