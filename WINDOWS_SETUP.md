# Getting set up on Windows — plain-language guide

This is the simple version. For the technical reference (exact contracts, exit codes,
troubleshooting detail), see `collection/README_ACQUISITION.md` and `TESTING.md`.

## What you need before you start

- [ ] The Unicorn Hybrid Black headset
- [ ] Its USB charging cable
- [ ] Its Bluetooth dongle (a small USB stick, ships in the box)
- [ ] A Windows PC with an internet connection and a free USB port
- [ ] This project's code on that PC (copy the whole folder over, or `git clone` it)

---

## Part 1 — Charge and prep the headset

1. Charge the headset with its USB charging cable (~3 hours from empty). The light
   turns off when it's full.
2. Clip the 8 electrodes into the marked holes in the cap.
3. Clip the 2 sticky reference electrodes behind your ears (on the bony part behind
   the earlobe).
4. Don't turn it on yet — that comes later, once the software is ready.

## Part 2 — Install the software this project needs

1. Install **Miniconda** for Windows: <https://docs.conda.io/en/latest/miniconda.html>
   (just click through the installer with defaults).
2. Open **Git Bash** (installed with Git for Windows — if you don't have it,
   install Git for Windows first). Everything below runs in this window, not in
   PowerShell or the regular Command Prompt.
3. In Git Bash, go into the project folder and run the setup script:
   ```bash
   cd "/path/to/P300 Speller"
   bash setup_windows.sh --test
   ```
   This creates the Python environment, installs everything the project needs, and
   (with `--test`) runs the full test suite as a final check. It takes a few minutes.
   You should see a wall of `OK` lines and, at the end, something like
   `676 passed, 3 skipped`. If anything says `FAIL`, stop here and fix that one
   thing before moving on — don't continue with a half-working environment. A `WARN`
   line about `torch` not importing is not fatal — it only affects the Transformer
   LM (`--lm-transformer-checkpoint`); uniform and ngram work regardless.

## Part 3 — Install g.tec's software and connect the headset

1. Install **Unicorn Suite** (g.tec's own software — comes on a USB stick with the
   headset, or download from g.tec's site).
2. Plug the **Bluetooth dongle** into a USB port on the PC. (This is the only cable
   that connects to the computer — the headset itself is wireless.)
3. Turn the headset on: hold its power button for 2 seconds.
4. Open **Unicorn Suite → "My Unicorn" tab**. Your headset should show up as
   unpaired — click it and follow the Windows pairing pop-up. A green light/circle
   means it's paired.
5. Open the **Unicorn LSL Interface** app (also part of Unicorn Suite) and click
   **Start**. It should say **"Streaming"** — if it only says "Connected," it's not
   actually sending data yet.

## Part 4 — Try it with no headset first (demo mode)

Before doing anything with real electrodes on your head, confirm the software itself
works:
```bash
bash start_speller.sh --demo
```
A window should open showing a Thai character grid, live (fake) EEG traces, and a
scalp map with green dots. This proves the software side works, independent of the
headset. Close it when you're done looking.

## Part 5 — Run it for real

1. Make sure the Unicorn LSL Interface (Part 3, step 5) is still running and says
   "Streaming."
2. In Git Bash:
   ```bash
   bash start_speller.sh
   ```
3. Watch the scalp map. Every dot should turn green within a few seconds — that
   means real electrode contact is good. If a dot stays red or gray, that
   electrode isn't making good contact; re-seat it.
4. Type the characters you want to calibrate on into "Target characters" and press
   **Start**.

That's the whole loop: charge → pair → LSL Interface streaming → `start_speller.sh`.
Everything after this (fitting a model, running live inference, evaluating
accuracy) is the same regardless of OS — see `CLAUDE.md`'s "Running" section.

---

## If something goes wrong

- **`setup_windows.sh` fails on a specific package** — PsychoPy and PySide6 are the
  usual suspects on Windows. The error message will name the missing piece; it's
  almost always a missing Visual C++ Redistributable. Install the latest one from
  Microsoft and re-run the script.
- **Scalp map won't turn green** — check the Unicorn LSL Interface really says
  "Streaming," not just "Connected." If it does, check electrode contact.
- **Two Bluetooth devices / can't pair** — if your laptop has its own built-in
  Bluetooth, disable it in Windows Device Manager so it doesn't fight with the
  Unicorn dongle for the pairing.
- **Nothing above helps** — `TESTING.md`'s "Hardware-in-the-loop checks" (H1–H11)
  walk through diagnosing the connection step by step, with go/no-go criteria for
  each one.
