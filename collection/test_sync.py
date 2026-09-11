"""
Minimal PsychoPy + LSL example -- RUN THIS FIRST, before anything else.

Purpose: prove your recording chain works end to end BEFORE you build (or
trust) a real experiment on top of it. This flashes a single square 20
times and pushes an LSL marker at the exact moment of each flash. If you
open LabRecorder alongside this and record, you should see two synced
streams in the resulting .xdf file: your Unicorn EEG stream, and this
script's 'PsychoPyMarkers' stream, timestamped in the same clock.

WHY THIS MATTERS: catching a sync problem now, on a trivial 20-flash test,
costs you two minutes. Catching it after a real multi-hour Thai-character
data collection session costs you the whole session.

SETUP BEFORE RUNNING:
  1. Open the Unicorn LSL Interface app (from Unicorn Suite), connect to
     your headset, click Start. Confirm it's streaming.
  2. Open LabRecorder (separate LSL app -- https://github.com/labstreaminglayer/App-LabRecorder).
     It should auto-discover the Unicorn EEG stream.
  3. Run this script. LabRecorder should also discover 'PsychoPyMarkers'
     (may need to hit refresh in LabRecorder's stream list).
  4. Click Record in LabRecorder, THEN run this script's flashes (the
     countdown below gives you time to do this).
  5. Stop LabRecorder. Inspect the resulting .xdf -- confirm both streams
     are present and marker timestamps look sane (evenly spaced, matching
     ISI_SECONDS below).

pip install psychopy pylsl
"""

from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet
import time

N_FLASHES = 20
ISI_SECONDS = 1.0          # time between flash onsets
FLASH_DURATION = 0.1       # how long the square stays highlighted

# --- LSL marker stream setup ---
# channel_format='string' since markers are just labels here, not numeric
# data -- LSL timestamps each push_sample() call automatically, which is
# the actual sync mechanism; the string content is just metadata.
info = StreamInfo(
    name="PsychoPyMarkers", type="Markers",
    channel_count=1, nominal_srate=0,  # irregular rate -- markers are event-driven, not periodic
    channel_format="string", source_id="p300_speller_markers_v1",
)
outlet = StreamOutlet(info)

# --- PsychoPy window + stimulus ---
win = visual.Window(size=(800, 600), color="black", units="pix", fullscr=False)
square = visual.Rect(win, width=200, height=200, fillColor="grey", lineColor="grey")

print(f"Starting in 5 seconds -- start LabRecorder's recording now.")
core.wait(5.0)

clock = core.Clock()
for i in range(N_FLASHES):
    # draw background frame first (square not shown)
    win.flip()
    core.wait(ISI_SECONDS - FLASH_DURATION)

    # flash onset -- push the marker as close to the actual draw as possible
    square.draw()
    win.flip()
    outlet.push_sample([f"flash_{i}"])
    onset_t = clock.getTime()
    print(f"Flash {i} at t={onset_t:.4f}s")

    core.wait(FLASH_DURATION)

    if event.getKeys(keyList=["escape"]):
        break

win.close()
core.quit()