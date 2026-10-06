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
        "Application:Bayesian%20Stopping int StoppingRule= 1 1 0 1 "
            "// 0: mean score after all flashes, 1: Bayesian dynamic stopping (enumeration)",
        "Application:Bayesian%20Stopping float StoppingLambda= 0.7 0.7 0 1 "
            "// prior blend: p0 = lambda * LM prior + (1 - lambda) / characters",
        "Application:Bayesian%20Stopping float StoppingTheta= 0.9 0.9 0 1 // stop once the posterior max reaches this",
        "Application:Bayesian%20Stopping int StoppingTMin= 6 6 0 % // flashes before stopping is allowed",
        "Application:Bayesian%20Stopping int StoppingTMax= 120 120 1 % // forced stop after this many flashes",
        "Application:Bayesian%20Stopping float StoppingLLRClip= 8 8 0 % // clip on each flash's log-likelihood ratio",
        "Application:Language%20Model int LanguageModel= 0 0 0 1 "
            "// 0: uniform prior, 1: Thai Transformer (LanguageModelFile); the fitted model's .prm sets it (enumeration)",
        "Application:Language%20Model string LanguageModelFile= % % % % "
            "// .lmbin written by pipeline/bci2000_lm_export.py (absolute path)",
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
        "PosteriorMax 32 0 0 0",    // float32 bits: the posterior maximum after the latest update
        "PosteriorArgmax 7 0 0 0",  // 1-based character with that maximum
        "StoppingFlashes 16 0 0 0", // flashes the controller has used for this character
        "StopReason 2 0 0 0",       // 1 threshold, 2 t_max, 3 flashes ran out (argmax selected)
        "SpellerRunID 31 0 0 0",    // random per run; the .applog's "RUN <id>" line starts that run's entries
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

    // Bayesian stopping
    const bool bayesian = Parameter("StoppingRule") == 1;
    double lam = Parameter("StoppingLambda"), theta = Parameter("StoppingTheta"), clip = Parameter("StoppingLLRClip");
    int tMin = Parameter("StoppingTMin"), tMax = Parameter("StoppingTMax");
    if (bayesian)
    {
        if (!(lam >= 0 && lam <= 1) || !(theta > 0 && theta < 1) || !(clip > 0) || tMax < 1 || tMin < 0 || tMin > tMax)
            bcierr << "Bayesian stopping needs 0 <= StoppingLambda <= 1, 0 < StoppingTheta < 1, StoppingLLRClip > 0 "
                   << "and 0 <= StoppingTMin <= StoppingTMax, StoppingTMax >= 1";
        // The controller needs Platt-calibrated scores: CheckerboardSignalProcessing's model provides a and b.
        // With no classifier at all (DummySignalProcessing, for calibration) nothing is ever selected.
        if (!Parameters->Exists("RiemannPlattA") && Parameters->Exists("Classifier"))
            bcierr << "Bayesian stopping needs CheckerboardSignalProcessing (Platt-calibrated scores); "
                   << "with another classifier set StoppingRule to 0";
    }
    // Initialize reads the calibration whenever the classifier provides it, so preflight must too
    // (BCI2000 refuses a parameter first read during initialization).
    if (Parameters->Exists("RiemannPlattA"))
    {
        Parameter("RiemannPlattA");
        Parameter("RiemannPlattB");
    }
    if (Parameter("LanguageModel") == 1)
    {
        ThaiGPT::Model lm;
        std::string file = Parameter("LanguageModelFile");
        std::string err = file.empty() ? "LanguageModelFile is empty" : lm.Load(file);
        if (!err.empty())
            bcierr << "Language model: " << err;
        else if (defs->NumColumns() >= 2)
        {
            // A checkpoint is valid only for the matrix it was trained against (CLAUDE.md): every matrix
            // character must be one code point of its vocabulary, in the order the file was checked against.
            std::vector<char32_t> matrix;
            for (int i = 0; i < n; ++i)
            {
                std::vector<char32_t> cps = ThaiGPT::DecodeUtf8(std::string(defs(i, 1)));
                if (cps.size() != 1 || lm.Id(cps[0]) < 0)
                    bcierr << "Language model: matrix character " << i + 1 << " (" << std::string(defs(i, 1))
                           << ") is not in the model's vocabulary; retrain against this matrix";
                matrix.push_back(cps.empty() ? 0 : cps[0]);
            }
            if (matrix != lm.matrix)
                bcierr << "Language model: the file was exported for a different matrix than TargetDefinitions";
        }
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

    mBayesian = Parameter("StoppingRule") == 1;
    mController = BayesianStopping::Controller();
    mController.nChars = Parameter("TargetDefinitions")->NumRows();
    mController.lam = Parameter("StoppingLambda");
    mController.theta = Parameter("StoppingTheta");
    mController.tMin = Parameter("StoppingTMin");
    mController.tMax = Parameter("StoppingTMax");
    mController.llrClip = Parameter("StoppingLLRClip");
    if (Parameters->Exists("RiemannPlattA")) // the classifier's calibration (pipeline/fit_from_recording.py)
    {
        mController.sigmoidA = Parameter("RiemannPlattA");
        mController.sigmoidB = Parameter("RiemannPlattB");
    }
    mpLanguageModel.reset();
    mMatrixTokenIds.clear();
    if (Parameter("LanguageModel") == 1)
    {
        mpLanguageModel.reset(new ThaiGPT::Model);
        mpLanguageModel->Load(Parameter("LanguageModelFile")); // checked in preflight
        ParamRef defs = Parameter("TargetDefinitions");
        for (int i = 0; i < defs->NumRows(); ++i)
            mMatrixTokenIds.push_back(mpLanguageModel->Id(ThaiGPT::DecodeUtf8(std::string(defs(i, 1)))[0]));
        bciout << "Language model: Thai Transformer " << mpLanguageModel->checkpointSha.substr(0, 12) << ", "
               << mpLanguageModel->nLayer << " layers, context " << mpLanguageModel->blockSize << " characters";
    }

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
    // Ties this run's .dat (the SpellerRunID state) to its entries in the session's .applog, which
    // BCI2000 shares between all runs of a session. std::random_device, not RandomNumberGenerator,
    // so the flash shuffle stays reproducible from RandomSeed.
    const uint32_t runId = 1 + std::random_device()() % 0x7FFFFFFE;
    State("SpellerRunID") = runId;
    AppLog.File << "RUN " << runId << std::endl;
    mStopped = mEndEarly = false;
    mDecision = -1;
    mStopReason = mPresented = 0;
    State("PosteriorMax") = 0;
    State("PosteriorArgmax") = 0;
    State("StoppingFlashes") = 0;
    State("StopReason") = 0;
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
    mPresented = 0;
    mStopped = mEndEarly = false;
    mDecision = -1;
    mStopReason = 0;
    if (mBayesian)
    {
        // The prior of the text spelled so far, as the Python service's get_prior(committed_text).
        std::vector<double> prior = CurrentPrior();
        mController.Reset(prior);
        SetPosteriorStates();
        State("StopReason") = 0;
        std::vector<int> order(mNumChars);
        for (int i = 0; i < mNumChars; ++i)
            order[i] = i;
        std::partial_sort(order.begin(), order.begin() + std::min(5, mNumChars), order.end(),
                          [&](int a, int b) { return prior[a] > prior[b]; });
        std::ostringstream top, full;
        for (int k = 0; k < std::min(5, mNumChars); ++k)
            top << " " << std::string(Parameter("TargetDefinitions")(order[k], 0)) << " " << prior[order[k]];
        full.precision(9);
        for (double p : prior)
            full << " " << p;
        AppLog << "Selection " << mSelections + 1 << " prior (" << (mpLanguageModel ? "Transformer" : "uniform")
               << ", context \"" << mText << "\"): top" << top.str() << std::endl;
        AppLog.File << "PRIOR " << mSelections + 1 << full.str() << std::endl;
    }
}

std::vector<double> CheckerboardSpellerTask::CurrentPrior() const
{
    if (mpLanguageModel)
        return mpLanguageModel->Prior(mText, mMatrixTokenIds);
    return std::vector<double>(mNumChars, 1.0 / mNumChars);
}

void CheckerboardSpellerTask::SetPosteriorStates()
{
    StateRef maxState = State("PosteriorMax");
    maxState.AsFloat() = static_cast<float>(mController.MaxProbability());
    State("PosteriorArgmax") = mController.Argmax() + 1;
    State("StoppingFlashes") = mController.t;
}

void CheckerboardSpellerTask::OnClassInput(int stimulusCode, const GenericSignal &signal)
{
    if (!mBayesian || mStopped || stimulusCode < 1 || stimulusCode > int(mGroups.size()))
        return;
    double s = 0;
    int n = 0;
    for (int ch = 0; ch < signal.Channels(); ++ch)
        for (int el = 0; el < signal.Elements(); ++el, ++n)
            s += signal(ch, el);
    s /= std::max(n, 1);
    const Checkerboard::Group &group = mGroups[stimulusCode - 1];
    const double pi = static_cast<double>(group.size()) / mNumChars; // as the service: len(group) / n_chars
    int idx = -1;
    const bool stop = mController.Update(group, s, pi, idx);
    SetPosteriorStates();
    if (stop)
    {
        mStopped = true;
        mEndEarly = true;
        mDecision = idx;
        const bool threshold = mController.t >= mController.tMin && mController.MaxProbability() >= mController.theta;
        mStopReason = threshold ? 1 : 2;
        State("StopReason") = mStopReason;
        AppLog << "Selection " << mSelections + 1 << ": stopped after " << mController.t << " flash result(s) ("
               << (threshold ? "threshold" : "t_max") << ", posterior " << mController.MaxProbability() << ", "
               << mPresented << " flashes shown)" << std::endl;
    }
}

int CheckerboardSpellerTask::OnNextStimulusCode()
{
    // A zero code ends the current selection; a zero right after a zero ends the run.
    // Bayesian stopping ends the selection early: no more flashes once the controller has decided.
    if (mInSelection && (mNextFlash == mGroups.size() || mEndEarly))
    {
        mInSelection = false;
        mEndEarly = false;
        return 0;
    }
    if (!mInSelection)
    {
        if (mInterpretMode_ == InterpretModes::Copy && mCueIndex >= mCueSequence.size())
            return 0; // everything cued: end the run
        StartSelection();
    }
    ++mPresented;
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
    size_t epochs = 0;
    Target *pTarget = nullptr;
    std::string rule;
    if (mBayesian)
    {
        for (const auto &r : inResult)
            if (Associations().find(r.first) != Associations().end())
                epochs += r.second.size();
        if (mStopped)
            pTarget = mTargets[mDecision];
        else if (mController.t >= 1)
        {
            // The flashes ran out before the stopping rule fired: the argmax, as the Python service's
            // trial-end commit.
            mDecision = mController.Argmax();
            mStopReason = 3;
            State("StopReason") = mStopReason;
            pTarget = mTargets[mDecision];
        }
        rule = mStopReason == 1 ? "threshold" : mStopReason == 2 ? "t_max" : mStopReason == 3 ? "flashes ran out"
                                                                                          : "no evidence";
    }
    else
    {
        pTarget = SelectByMeanScore(inResult, epochs);
        rule = "mean score";
    }
    mHadResults = true;
    int selected = pTarget ? pTarget->Tag() : 0;
    State("SelectedChar") = selected;
    AppLog << "Selection " << mSelections + 1 << ": " << epochs << " flash result(s) for " << mPresented
           << " flashes; selected "
           << (selected ? std::string(Parameter("TargetDefinitions")(selected - 1, 0)) : std::string("nothing"))
           << " (" << rule << (mBayesian ? ", used " + std::to_string(mController.t) : std::string()) << ")"
           << (mAttendedChar >= 0
                   ? ", cued " + std::string(Parameter("TargetDefinitions")(mAttendedChar, 0))
                   : std::string())
           << std::endl;
    return pTarget;
}

Target *CheckerboardSpellerTask::SelectByMeanScore(const ClassResult &inResult, size_t &epochs)
{
    // Each character's evidence is the MEAN of the scores of the flashes that contained it and were
    // scored. StimulusTask's default sums them, and a flash the classifier rejected (lost samples, a
    // flat channel) then counts as 0 -- not neutral for an LDA score, whose non-target flashes sit
    // around log(1/11) or lower -- which would push the characters of rejected flashes up. A character
    // with no scored flash at all cannot be selected.
    std::vector<double> sum(mNumChars, 0.0);
    std::vector<int> count(mNumChars, 0);
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
