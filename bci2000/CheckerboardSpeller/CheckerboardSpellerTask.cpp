////////////////////////////////////////////////////////////////////////////////
// CheckerboardSpellerTask: see CheckerboardSpellerTask.h.
////////////////////////////////////////////////////////////////////////////////
#include "CheckerboardSpellerTask.h"

#include "BCIStream.h"

#include <algorithm>
#include <random>
#include <sstream>
#include <vector>

RegisterFilter(CheckerboardSpellerTask, 3);

namespace
{
const int NumFlashStates = 6;     // characters per checkerboard group (6 x 6 virtual matrices)
const int MaxCharsInStates = 127; // character states are 7 bits, 1-based

// Spelling check with no side effects, for preflight.
struct PreflightSpeller : Speller
{
    ~PreflightSpeller()
    {
        DeleteObjects();
    }
    void OnEnter(const std::string &) override
    {
    }
};
} // namespace

////////////////////////////////////////////////////////////////////////////////
// CheckerboardCell

CheckerboardCell &CheckerboardCell::SetColors(RGBColor background, RGBColor text, RGBColor flashBackground,
                                              RGBColor flashText, RGBColor cue)
{
    mBackground = background;
    mText = text;
    mFlashBackground = flashBackground;
    mFlashText = flashText;
    mCueColor = cue;
    SetColor(BeingPresented() ? mFlashBackground : mBackground);
    Invalidate();
    return *this;
}

void CheckerboardCell::OnPresent()
{
    SetColor(mFlashBackground);
    TextStimulus::OnPresent();
}

void CheckerboardCell::OnConceal()
{
    SetColor(mBackground);
    TextStimulus::OnConceal();
}

CheckerboardCell &CheckerboardCell::SetCue(bool cue)
{
    if (cue != mCue)
    {
        mCue = cue;
        Invalidate();
    }
    return *this;
}

void CheckerboardCell::OnPaint(const GUI::DrawContext &dc)
{
    // The fill comes from Color (set in OnPresent/OnConceal); the second colour argument is unused.
    if (BeingPresented())
        TextField::DoPaint(dc, mFlashText, mFlashBackground);
    else
        TextField::DoPaint(dc, mCue ? mCueColor : mText, mBackground);
}

////////////////////////////////////////////////////////////////////////////////
// CheckerboardSpellerTask

CheckerboardSpellerTask::CheckerboardSpellerTask()
{
    BEGIN_PARAMETER_DEFINITIONS
        "Application:Sequencing int NumberOfSequences= 5 5 1 % "
            "// repetitions per selection; each repetition flashes every character twice",
        "Application:Speller%20Targets matrix TargetDefinitions= "
            "6 {Display Enter} A A B B C C D D E E F F "
            "// matrix characters, row-major; load parms/ThaiCheckerboard.prm for the Thai matrix",
        "Application:Speller%20Targets int NumMatrixRows= 2 8 1 % // rows of the display matrix",
        "Application:Speller%20Targets int NumMatrixColumns= 3 9 1 % "
            "// columns of the display matrix; must be odd for the checkerboard to be adjacency-safe",
        "Application:Speller%20Targets float CellFill= 0.9 0.9 0.1 1 "
            "// fraction of each grid cell covered by the flash box",
        "Application:Speller%20Targets float TargetTextHeight= 6 6 0 100 "
            "// height of matrix characters in percent of screen height",
        "Application:Speller%20Targets string BackgroundColor= 0x000000 0x000000 % % // background (color)",
        "Application:Speller%20Targets string TextColor= 0xDCDCDC 0xDCDCDC % % // idle text (color)",
        "Application:Speller%20Targets string IntensifiedBackgroundColor= 0xFFFFFF 0xFFFFFF % % "
            "// flash box (color)",
        "Application:Speller%20Targets string IntensifiedTextColor= 0x000000 0x000000 % % "
            "// text on the flash box (color)",
        "Application:Speller%20Targets string CueColor= 0x00FF00 0x00FF00 % % "
            "// cued character's text during the pre-sequence pause, copy mode (color)",
        "Application:Speller float StatusBarSize= 10 10 0 100 // status bar height in percent of screen height",
        "Application:Speller float StatusBarTextHeight= 6 6 0 100 "
            "// status bar text height in percent of screen height",
        "Application:Speller string TextToSpell= % % % % // text to spell in copy mode, one cue per character",
        "Application:Speller string TextResult= % % % % // spelling result",
    END_PARAMETER_DEFINITIONS

    BEGIN_STATE_DEFINITIONS
        "FlashChar1 7 0 0 0",
        "FlashChar2 7 0 0 0",
        "FlashChar3 7 0 0 0",
        "FlashChar4 7 0 0 0",
        "FlashChar5 7 0 0 0",
        "FlashChar6 7 0 0 0",
        "AttendedChar 7 0 0 0",
        "SelectedChar 7 0 0 0",
    END_STATE_DEFINITIONS
}

CheckerboardSpellerTask::~CheckerboardSpellerTask()
{
    ClearObjects();
}

void CheckerboardSpellerTask::ClearObjects()
{
    Associations().clear();
    for (CheckerboardCell *p : mCells)
        delete p;
    mCells.clear();
    Speller::DeleteObjects();
    mTargets.clear();
}

void CheckerboardSpellerTask::OnPreflight(const SignalProperties &) const
{
    Parameter("TextResult");
    Parameter("StatusBarSize");
    Parameter("StatusBarTextHeight");
    Parameter("CellFill");
    Parameter("TargetTextHeight");
    const char *colors[] = {"BackgroundColor", "TextColor", "IntensifiedBackgroundColor", "IntensifiedTextColor",
                            "CueColor"};
    for (const char *c : colors)
        RGBColor(Parameter(c));

    int rows = Parameter("NumMatrixRows"), cols = Parameter("NumMatrixColumns");
    ParamRef defs = Parameter("TargetDefinitions");
    int n = defs->NumRows();
    if (defs->NumColumns() < 2)
        bcierr << "TargetDefinitions needs Display and Enter columns";
    if (n != rows * cols)
        bcierr << "TargetDefinitions has " << n << " entries; NumMatrixRows x NumMatrixColumns is " << rows * cols;
    if (n % 2 != 0)
        bcierr << "The checkerboard needs an even number of characters, not " << n;
    if (n > MaxCharsInStates)
        bcierr << "At most " << MaxCharsInStates << " characters fit the FlashChar states";
    if (n == rows * cols && n % 2 == 0 && n > 0)
    {
        std::pair<int, int> shape = Checkerboard::VirtualShape(n / 2);
        if (shape.first > NumFlashStates || shape.second > NumFlashStates)
            bcierr << "Flash groups of up to " << NumFlashStates << " characters fit the FlashChar states; "
                   << n << " characters make groups of " << std::max(shape.first, shape.second);
        // Adjacency safety holds for every shuffle or none (it depends on parity and grid
        // width only), so one sequence decides it.
        std::mt19937 rng(1);
        auto violations = Checkerboard::AdjacentCoFlashes(Checkerboard::BuildFlashSequence(n, 1, rng), rows, cols);
        if (!violations.empty())
            bcierr << "Neighbouring characters " << violations[0].first + 1 << " and " << violations[0].second + 1
                   << " can flash together on a " << rows << " x " << cols
                   << " grid; the checkerboard needs an odd number of columns";
    }
    for (int i = 0; i < n && defs->NumColumns() >= 2; ++i)
        if (std::string(defs(i, 1)).empty())
            bcierr << "TargetDefinitions entry " << i + 1 << " has empty Enter text";

    // Results are scored per selection (OnClassResult): evidence carried over to a next selection would
    // be credited to whichever characters reuse those flash numbers there.
    if (Parameter("AccumulateEvidence") != 0)
        bcierr << "AccumulateEvidence must be 0: flash numbers are reused by every selection";
    if (Parameter("MinimumEvidence") != 0)
        bcierr << "MinimumEvidence must be 0: this speller selects by the mean score per character";
    // Every flash has its own stimulus code, so the classifier must report each epoch on its own.
    // EpochsToAverage = 1 is not enough: without SingleEpochMode, P3TemporalFilter keeps counting
    // a code's epochs across selections, reports only when the count equals EpochsToAverage and
    // resets at NumberOfSequences, so every other selection would get no results at all.
    if (OptionalParameter("EpochsToAverage", 1) != 1)
        bcierr << "EpochsToAverage must be 1: each checkerboard flash has its own stimulus code";
    if (OptionalParameter("SingleEpochMode", 1) != 1)
        bcierr << "SingleEpochMode must be 1: each checkerboard flash has its own stimulus code, and "
               << "P3TemporalFilter would otherwise drop the results of every other selection";

    if (Parameter("InterpretMode") == InterpretModes::Copy)
    {
        std::string text = Parameter("TextToSpell");
        if (text.empty())
            bcierr << "Copy mode needs TextToSpell";
        PreflightSpeller speller;
        for (int i = 0; i < n && defs->NumColumns() >= 2; ++i)
            (new SpellerTarget(speller))->SetEntryText(defs(i, 1));
        if (!speller.TrySpelling(text))
            bcierr << "TextToSpell contains characters that are not in the matrix";
    }
}

void CheckerboardSpellerTask::OnInitialize(const SignalProperties &)
{
    ClearObjects();
    mNumRows = Parameter("NumMatrixRows");
    mNumCols = Parameter("NumMatrixColumns");
    mNumberOfSequences = Parameter("NumberOfSequences");
    mInterpretMode_ = Parameter("InterpretMode");
    mTextToSpell = mInterpretMode_ == InterpretModes::Copy ? std::string(Parameter("TextToSpell")) : "";

    if (mpStatusBar == nullptr)
        mpStatusBar = new StatusBar(Display());
    float barSize = Parameter("StatusBarSize") / 100.0f;
    GUI::Rect barRect = {0, 0, 1.0f, barSize};
    mpStatusBar->SetTextHeight(Parameter("StatusBarTextHeight") / Parameter("StatusBarSize"))
        .SetColor(RGBColor::Gray)
        .SetTextColor(RGBColor::Yellow)
        .SetObjectRect(barRect);
    mpStatusBar->SetGoalText(mTextToSpell);
    mpStatusBar->SetResultText("");

    // Grid below the status bar, cells evenly spaced as in the Python stimulus.
    ParamRef defs = Parameter("TargetDefinitions");
    mNumChars = defs->NumRows();
    float fill = Parameter("CellFill"), cellW = 1.0f / mNumCols, cellH = (1.0f - barSize) / mNumRows;
    float textHeight = Parameter("TargetTextHeight") / 100.0f / (cellH * fill);
    RGBColor bg(Parameter("BackgroundColor")), text(Parameter("TextColor")),
        flashBg(Parameter("IntensifiedBackgroundColor")), flashText(Parameter("IntensifiedTextColor")),
        cue(Parameter("CueColor"));
    for (int i = 0; i < mNumChars; ++i)
    {
        int r = i / mNumCols, c = i % mNumCols;
        float cx = (c + 0.5f) * cellW, cy = barSize + (r + 0.5f) * cellH;
        GUI::Rect rect = {cx - cellW * fill / 2, cy - cellH * fill / 2, cx + cellW * fill / 2, cy + cellH * fill / 2};
        CheckerboardCell *pCell = new CheckerboardCell(Display());
        pCell->SetColors(bg, text, flashBg, flashText, cue);
        pCell->SetText(defs(i, 0)).SetTextHeight(textHeight).SetObjectRect(rect);
        pCell->SetPresentationMode(VisualStimulus::Intensify);
        mCells.push_back(pCell);

        SpellerTarget *pTarget = new SpellerTarget(*this);
        pTarget->SetEntryText(defs(i, 1)).SetTag(i + 1);
        mTargets.push_back(pTarget);
    }
}

void CheckerboardSpellerTask::OnStartRun()
{
    mText = std::string(Parameter("TextResult"));
    mpStatusBar->SetResultText(mText);
    mCueSequence.clear();
    mCueIndex = 0;
    mAttendedChar = -1;
    mInSelection = false;
    mNextFlash = 0;
    mGroups.clear();
    mSelections = 0;
    mHadResults = false;
    if (mInterpretMode_ == InterpretModes::Copy)
    {
        SequenceOfSpellerTargets seq;
        TrySpelling(mTextToSpell, &seq);
        for (SpellerTarget *p : seq)
            if (p)
                mCueSequence.push_back(p->Tag() - 1);
    }
    for (int s = 1; s <= NumFlashStates; ++s)
        State("FlashChar" + std::to_string(s)) = 0;
    State("AttendedChar") = 0;
    State("SelectedChar") = 0;
    AppLog << "Checkerboard speller: " << mNumRows << " x " << mNumCols << ", " << mNumberOfSequences
           << " repetition(s) per selection"
           << (mInterpretMode_ == InterpretModes::Copy ? ", copy mode: " + mTextToSpell : std::string())
           << std::endl;
    DisplayMessage("Waiting to start ...");
}

void CheckerboardSpellerTask::StartSelection()
{
    mGroups = Checkerboard::BuildFlashSequence(mNumChars, mNumberOfSequences, RandomNumberGenerator);
    Associations().clear();
    for (size_t k = 0; k < mGroups.size(); ++k)
    {
        Association &a = Associations()[static_cast<int>(k + 1)];
        for (int idx : mGroups[k])
        {
            a.Add(mCells[idx]);
            a.Add(mTargets[idx]);
        }
    }
    mAttendedChar = -1;
    if (mInterpretMode_ == InterpretModes::Copy && mCueIndex < mCueSequence.size())
        mAttendedChar = mCueSequence[mCueIndex];
    SetAttendedTarget(mAttendedChar >= 0 ? mTargets[mAttendedChar] : nullptr);
    mNextFlash = 0;
    mInSelection = true;
}

int CheckerboardSpellerTask::OnNextStimulusCode()
{
    // A zero code ends the current selection; a zero right after a zero ends the run.
    if (mInSelection && mNextFlash == mGroups.size())
    {
        mInSelection = false;
        return 0;
    }
    if (!mInSelection)
    {
        if (mInterpretMode_ == InterpretModes::Copy && mCueIndex >= mCueSequence.size())
            return 0; // everything cued: end the run
        StartSelection();
    }
    return static_cast<int>(++mNextFlash);
}

void CheckerboardSpellerTask::OnPreSequence()
{
    DisplayMessage("");
    State("AttendedChar") = mAttendedChar + 1;
    State("SelectedChar") = 0;
    if (mAttendedChar >= 0)
        mCells[mAttendedChar]->SetCue(true);
}

void CheckerboardSpellerTask::OnSequenceBegin()
{
    if (mAttendedChar >= 0)
        mCells[mAttendedChar]->SetCue(false);
}

void CheckerboardSpellerTask::SetFlashStates(int stimulusCode)
{
    const Checkerboard::Group *g =
        stimulusCode > 0 && stimulusCode <= int(mGroups.size()) ? &mGroups[stimulusCode - 1] : nullptr;
    for (int s = 0; s < NumFlashStates; ++s)
        State("FlashChar" + std::to_string(s + 1)) = (g && s < int(g->size())) ? (*g)[s] + 1 : 0;
}

void CheckerboardSpellerTask::OnStimulusBegin(int stimulusCode)
{
    SetFlashStates(stimulusCode);
    StimulusTask::OnStimulusBegin(stimulusCode);
}

void CheckerboardSpellerTask::OnStimulusEnd(int stimulusCode)
{
    StimulusTask::OnStimulusEnd(stimulusCode);
    SetFlashStates(0);
}

Target *CheckerboardSpellerTask::OnClassResult(const ClassResult &inResult)
{
    // Each character's evidence is the MEAN of the scores of the flashes that contained it and were
    // scored. StimulusTask's default sums them, and a flash the classifier rejected (lost samples, a
    // flat channel) then counts as 0 -- not neutral for an LDA score, whose non-target flashes sit
    // around log(1/11) or lower -- which would push the characters of rejected flashes up. A character
    // with no scored flash at all cannot be selected. (Bayesian stopping will replace this rule.)
    std::vector<double> sum(mNumChars, 0.0);
    std::vector<int> count(mNumChars, 0);
    size_t epochs = 0;
    for (const auto &r : inResult)
    {
        auto a = Associations().find(r.first);
        if (a == Associations().end() || r.second.empty())
            continue;
        double flashScore = 0;
        size_t n = 0;
        for (const GenericSignal &sig : r.second)
            for (int ch = 0; ch < sig.Channels(); ++ch)
                for (int el = 0; el < sig.Elements(); ++el, ++n)
                    flashScore += sig(ch, el);
        flashScore /= std::max<size_t>(n, 1);
        epochs += r.second.size();
        for (Target *t : a->second.Targets())
        {
            const int idx = t->Tag() - 1;
            if (idx >= 0 && idx < mNumChars)
            {
                sum[idx] += flashScore;
                ++count[idx];
            }
        }
    }
    Target *pTarget = nullptr;
    double best = 0;
    for (int i = 0; i < mNumChars; ++i)
        if (count[i] > 0 && (pTarget == nullptr || sum[i] / count[i] > best))
        {
            best = sum[i] / count[i];
            pTarget = mTargets[i];
        }
    mHadResults = true;
    int selected = pTarget ? pTarget->Tag() : 0;
    State("SelectedChar") = selected;
    AppLog << "Selection " << mSelections + 1 << ": " << epochs << " flash result(s) for " << mGroups.size()
           << " flashes; selected "
           << (selected ? std::string(Parameter("TargetDefinitions")(selected - 1, 0)) : std::string("nothing"))
           << (mAttendedChar >= 0
                   ? ", cued " + std::string(Parameter("TargetDefinitions")(mAttendedChar, 0))
                   : std::string())
           << std::endl;
    return pTarget;
}

void CheckerboardSpellerTask::OnSequenceEnd()
{
    // StimulusTask also calls this when a run is stopped mid-selection; that selection simply ended early.
    if (mInSelection)
        AppLog << "Selection " << mSelections + 1 << ": stopped before its last flash" << std::endl;
    else if (!mHadResults && mInterpretMode_ != InterpretModes::None)
        AppLog << "Selection " << mSelections + 1 << ": no flash results arrived; nothing selected" << std::endl;
    mHadResults = false;
    ++mSelections;
    if (mInterpretMode_ == InterpretModes::Copy)
        ++mCueIndex; // the next cue comes whatever was selected, as in calibration
    State("AttendedChar") = 0;
}

void CheckerboardSpellerTask::OnEnter(const std::string &inText)
{
    mText += inText;
    mpStatusBar->SetResultText(mText);
}

void CheckerboardSpellerTask::OnPostRun()
{
    SetFlashStates(0);
    State("SelectedChar") = 0;
}

void CheckerboardSpellerTask::OnStopRun()
{
    if (mAttendedChar >= 0 && mAttendedChar < int(mCells.size()))
        mCells[mAttendedChar]->SetCue(false);
    Parameter("TextResult") = mText;
    AppLog << "Run ended after " << mSelections << " selection(s); text: " << mText << std::endl;
    DisplayMessage("");
}
