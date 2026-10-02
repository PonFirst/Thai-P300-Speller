////////////////////////////////////////////////////////////////////////////////
// CheckerboardSpellerTask: BCI2000 application module for the Thai checkerboard
// P300 speller (Pananookooln et al. 2022 paradigm).
//
// Built on BCI2000's StimulusTask, like the stock P3Speller, and reusing its shared
// pieces (Speller targets, StatusBar, AppLog), but with checkerboard flash groups
// instead of rows and columns. What differs from P3Speller, and why:
//   - Groups are reshuffled every repetition (CheckerboardSequence.h, a port of
//     collection/flash_sequence.py), so stimulus codes cannot mean fixed rows and
//     columns. Each selection gets fresh codes 1..24*NumberOfSequences, one per
//     flash, and the associations are rebuilt for it. The classifier must therefore
//     report one result per flash (P3TemporalFilter: EpochsToAverage = 1).
//   - The recording carries its own flash metadata: states FlashChar1..FlashChar6
//     hold the 1-based indices of the characters lit right now (0 when nothing is
//     lit), AttendedChar the cued character in copy mode, SelectedChar the result.
//     With StimulusCode/StimulusType/StimulusBegin this is everything the old marker
//     JSON carried, inside the .dat file.
//   - Copy mode cues each character of TextToSpell in turn (lit in CueColor during
//     the pre-sequence pause), and moves to the next one after every selection
//     whatever was selected, as the Python stimulus did for calibration.
//   - A character's evidence is the mean score of its scored flashes, not StimulusTask's sum, so a
//     flash the classifier rejected does not bias the selection (see OnClassResult).
//   - Cells look like the Python stimulus: idle text on the background, and a
//     flashing cell inverted (IntensifiedBackgroundColor box, IntensifiedTextColor).
////////////////////////////////////////////////////////////////////////////////
#ifndef CHECKERBOARD_SPELLER_TASK_H
#define CHECKERBOARD_SPELLER_TASK_H

#include "CheckerboardSequence.h"
#include "Speller.h"
#include "StatusBar.h"
#include "StimulusTask.h"
#include "TextStimulus.h"

#include <string>
#include <vector>

// One matrix cell: a text stimulus that inverts when flashed and can show a cue.
class CheckerboardCell : public TextStimulus
{
  public:
    CheckerboardCell(GUI::GraphDisplay &display) : TextStimulus(display)
    {
    }
    CheckerboardCell &SetColors(RGBColor background, RGBColor text, RGBColor flashBackground, RGBColor flashText,
                                RGBColor cue);
    CheckerboardCell &SetCue(bool);
    bool Cue() const
    {
        return mCue;
    }

  protected:
    // TextField::DoPaint ignores the background colour it is given and fills the cell from the
    // field's own Color property (only when that is not NullColor), so the flash box is drawn by
    // switching Color here: the flash colour while presented, the background otherwise.
    void OnPresent() override;
    void OnConceal() override;
    void OnPaint(const GUI::DrawContext &) override;

  private:
    RGBColor mBackground = RGBColor::Black, mText = RGBColor::White, mFlashBackground = RGBColor::White,
             mFlashText = RGBColor::Black, mCueColor = RGBColor::Lime;
    bool mCue = false;
};

class CheckerboardSpellerTask : public StimulusTask, public Speller
{
  public:
    CheckerboardSpellerTask();
    ~CheckerboardSpellerTask();

  protected:
    // StimulusTask events
    void OnPreflight(const SignalProperties &Input) const override;
    void OnInitialize(const SignalProperties &Input) override;
    void OnStartRun() override;
    void OnStopRun() override;
    void OnPreSequence() override;
    void OnSequenceBegin() override;
    void OnSequenceEnd() override;
    void OnStimulusBegin(int stimulusCode) override;
    void OnStimulusEnd(int stimulusCode) override;
    void OnPostRun() override;
    Target *OnClassResult(const ClassResult &) override;
    int OnNextStimulusCode() override;
    // Speller events
    void OnEnter(const std::string &) override;

  private:
    void StartSelection();
    void ClearObjects();
    void SetFlashStates(int stimulusCode);

    // Configuration
    int mNumRows = 0, mNumCols = 0, mNumChars = 0, mNumberOfSequences = 0, mInterpretMode_ = 0;
    std::string mTextToSpell;

    // Display
    StatusBar *mpStatusBar = nullptr;
    std::vector<CheckerboardCell *> mCells;      // by character index
    std::vector<SpellerTarget *> mTargets;       // by character index

    // Copy mode: the characters to cue, in order, and the next one
    std::vector<int> mCueSequence;
    size_t mCueIndex = 0;
    int mAttendedChar = -1;

    // The current selection's flash groups; stimulus code k is mGroups[k - 1]
    std::vector<Checkerboard::Group> mGroups;
    size_t mNextFlash = 0;
    bool mInSelection = false;
    int mSelections = 0;
    bool mHadResults = false; // OnClassResult ran for the current selection

    std::string mText; // spelled so far
};

#endif // CHECKERBOARD_SPELLER_TASK_H
