////////////////////////////////////////////////////////////////////////////////
// RiemannP300Classifier: see RiemannP300Classifier.h.
////////////////////////////////////////////////////////////////////////////////
#include "RiemannP300Classifier.h"

#include "BCIStream.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

// Registered in PipeDefinition.cpp, which defines the module's filter chain.

namespace
{
const char *const ExportFormat = "p300-bci2000-riemann-v1"; // pipeline/bci2000_export.py EXPORT_FORMAT
const char *const NotLoaded = "none";
} // namespace

RiemannP300Classifier::RiemannP300Classifier()
{
    // The defaults describe no model; preflight refuses to run until an exported .prm is loaded.
    BEGIN_PARAMETER_DEFINITIONS
        "Filtering:Riemann%20P300%20Classifier string RiemannFormat= none none % % "
            "// export format; load the .prm written by pipeline/bci2000_export.py",
        "Filtering:Riemann%20P300%20Classifier string RiemannModelID= % % % % // sha256 of the fitted checkpoint",
        "Filtering:Riemann%20P300%20Classifier string RiemannProvenance= % % % % // what the model was fitted on",
        "Filtering:Riemann%20P300%20Classifier list RiemannChannels= 0 % % % // channel names, in order",
        "Filtering:Riemann%20P300%20Classifier float RiemannSamplingRate= 0 0 0 % // Hz",
        "Filtering:Riemann%20P300%20Classifier int RiemannEpochSamples= 0 0 0 % // samples from the flash onset",
        "Filtering:Riemann%20P300%20Classifier int RiemannPadSamples= 0 0 0 % // raw samples filtered on each side",
        "Filtering:Riemann%20P300%20Classifier intlist RiemannBaseline= 0 % % % // baseline sample range, or empty",
        "Filtering:Riemann%20P300%20Classifier matrix RiemannFilterSOS= 0 6 // band-pass second-order sections",
        "Filtering:Riemann%20P300%20Classifier matrix RiemannFilterZI= 0 2 // sosfilt_zi initial conditions",
        "Filtering:Riemann%20P300%20Classifier int RiemannFilterPadlen= 0 0 0 % // odd-extension length",
        "Filtering:Riemann%20P300%20Classifier matrix RiemannXdawnFilters= 0 0 // xDAWN spatial filters",
        "Filtering:Riemann%20P300%20Classifier matrix RiemannPrototypes= 0 0 // xDAWN prototypes",
        "Filtering:Riemann%20P300%20Classifier matrix RiemannRefInvSqrt= 0 0 // tangent-space reference^-1/2",
        "Filtering:Riemann%20P300%20Classifier floatlist RiemannLDAWeights= 0 % % % // LDA weights",
        "Filtering:Riemann%20P300%20Classifier float RiemannLDAIntercept= 0 0 % % // LDA intercept",
        "Filtering:Riemann%20P300%20Classifier float RiemannPlattA= 0 0 % % // Platt slope",
        "Filtering:Riemann%20P300%20Classifier float RiemannPlattB= 0 0 % % // Platt intercept",
        "Filtering:Riemann%20P300%20Classifier int RiemannQualityGate= 1 1 0 1 "
            "// reject a flash with a flat or railed channel, as the Python service (boolean)",
        "Filtering:Riemann%20P300%20Classifier float RiemannGateFlatUV= 1 1 0 % // flat below this peak-to-peak (uV)",
        "Filtering:Riemann%20P300%20Classifier float RiemannGateRailedUV= 400 400 0 % // railed above this (uV)",
        "Filtering:Riemann%20P300%20Classifier int RiemannGateSegmentSamples= 25 25 5 % // flat-segment length",
        // StimulusTask (the speller) reads EpochLength as the time from the last flash's onset until its
        // result is in, and decides then; results arriving later would be counted for the next character.
        "Filtering int EpochLength= 1400ms 1400ms 0 % "
            "// time from a flash's onset until its result is reported (at least epoch + pad + 2 blocks)",
    END_PARAMETER_DEFINITIONS

    BEGIN_STATE_DEFINITIONS
        "StimulusCodeRes 16 0 0 0", // as P3TemporalFilter: the flash a result belongs to
        "StimulusTypeRes 3 0 0 0",
        "ClassifierScore 32 0 0 0",        // float32 bits: the LDA decision score of that flash
        "ClassifierOnsetSample 32 0 0 0",  // the flash's onset, as a sample index within the run
        "ClassifierRejectedCode 16 0 0 0", // a flash that was not scored
    END_STATE_DEFINITIONS
}

std::string RiemannP300Classifier::LoadModel(RiemannP300::Model &m) const
{
    m = RiemannP300::Model();
    m.nChannels = Parameter("RiemannChannels")->NumValues();
    m.nTimes = Parameter("RiemannEpochSamples");
    m.nPad = Parameter("RiemannPadSamples");
    m.padlen = Parameter("RiemannFilterPadlen");
    ParamRef baseline = Parameter("RiemannBaseline");
    if (baseline->NumValues() == 2)
    {
        m.hasBaseline = true;
        m.baseline0 = baseline(0);
        m.baseline1 = baseline(1);
    }
    else if (baseline->NumValues() != 0)
        return "RiemannBaseline must be empty or two sample indices";

    ParamRef sos = Parameter("RiemannFilterSOS"), zi = Parameter("RiemannFilterZI");
    if (sos->NumRows() > 0 && sos->NumColumns() != 6)
        return "RiemannFilterSOS needs 6 columns";
    if (zi->NumRows() > 0 && zi->NumColumns() != 2)
        return "RiemannFilterZI needs 2 columns";
    for (int r = 0; r < sos->NumRows(); ++r)
    {
        std::array<double, 6> s;
        for (int c = 0; c < 6; ++c)
            s[c] = sos(r, c);
        m.sos.push_back(s);
    }
    for (int r = 0; r < zi->NumRows(); ++r)
        m.zi.push_back({double(zi(r, 0)), double(zi(r, 1))});

    auto matrix = [this](const char *name, Eigen::MatrixXd &out) {
        ParamRef p = Parameter(name);
        out.resize(p->NumRows(), p->NumColumns());
        for (int r = 0; r < p->NumRows(); ++r)
            for (int c = 0; c < p->NumColumns(); ++c)
                out(r, c) = p(r, c);
    };
    matrix("RiemannXdawnFilters", m.xdawn);
    matrix("RiemannPrototypes", m.prototypes);
    matrix("RiemannRefInvSqrt", m.refInvSqrt);
    ParamRef w = Parameter("RiemannLDAWeights");
    m.weights.resize(w->NumValues());
    for (int k = 0; k < w->NumValues(); ++k)
        m.weights(k) = w(k);
    m.intercept = Parameter("RiemannLDAIntercept");
    m.gate = Parameter("RiemannQualityGate") != 0;
    m.gateFlatUV = Parameter("RiemannGateFlatUV");
    m.gateRailedUV = Parameter("RiemannGateRailedUV");
    m.gateSegment = Parameter("RiemannGateSegmentSamples");

    std::string err = m.Check();
    if (!err.empty())
        return err;
    auto finite = [](const Eigen::MatrixXd &a) { return a.allFinite(); };
    if (!finite(m.xdawn) || !finite(m.prototypes) || !finite(m.refInvSqrt) || !m.weights.allFinite() ||
        !std::isfinite(m.intercept))
        return "the model contains non-finite values";
    return "";
}

void RiemannP300Classifier::Preflight(const SignalProperties &Input, SignalProperties &Output) const
{
    State("StimulusCode");
    State("StimulusType");
    OptionalState("Validation");
    OptionalState("Counter");
    Parameter("RiemannModelID");
    Parameter("RiemannProvenance");
    Parameter("RiemannPlattA");
    Parameter("RiemannPlattB");

    std::string format = Parameter("RiemannFormat");
    if (format == NotLoaded)
        bcierr << "No classifier loaded: load the .prm that pipeline/fit_from_recording.py wrote "
               << "(models/<id>_bci2000.prm) in Config";
    else if (format != ExportFormat)
        bcierr << "RiemannFormat is " << format << "; this filter reads " << ExportFormat;
    else
    {
        RiemannP300::Model m;
        std::string err = LoadModel(m);
        if (!err.empty())
            bcierr << "Classifier parameters: " << err;
        else
        {
            // Channels by name and order, as the Python service checks them: a classifier fitted on
            // one montage must never score another.
            ParamRef names = Parameter("RiemannChannels");
            if (Input.Channels() != m.nChannels)
                bcierr << "The classifier expects " << m.nChannels << " channels, the source sends "
                       << Input.Channels();
            else
                for (int c = 0; c < m.nChannels; ++c)
                    if (Input.ChannelLabels()[c] != std::string(names(c)))
                        bcierr << "Channel " << c + 1 << " is \"" << Input.ChannelLabels()[c]
                               << "\"; the classifier was fitted on \"" << std::string(names(c))
                               << "\". Set ChannelNames to the classifier's montage, in order";
            double fs = Parameter("SamplingRate").InHertz(), modelFs = Parameter("RiemannSamplingRate");
            if (std::fabs(fs - modelFs) > 1e-9)
                bcierr << "SamplingRate is " << fs << " Hz; the classifier was fitted at " << modelFs << " Hz";
            // The application sets StimulusCode in block a; this filter first sees it in block a + 1, at
            // the block's first sample (StimulusCode is constant within a block, so there is at most one
            // onset per block and windows complete in distinct blocks: results never queue). The window
            // ends n_times + pad - 1 samples later, so the result reaches the application in block
            // a + 1 + floor((n_times + pad - 1) / B), before its classification check in that block.
            // EpochLength must cover that; two blocks of margin on top.
            const int B = Input.Elements();
            const int needBlocks = 1 + (m.nTimes + m.nPad - 1) / B + 2;
            const double haveBlocks = Parameter("EpochLength").InSampleBlocks();
            if (haveBlocks < needBlocks)
                bcierr << "EpochLength must be at least " << std::ceil(needBlocks * B * 1000.0 / fs)
                       << " ms: each result is reported " << (m.nTimes + m.nPad) * 1000.0 / fs
                       << " ms after its flash (epoch plus zero-phase pad), and the speller would otherwise "
                       << "decide before the last results arrive and count them for the next character";
        }
    }
    // The model was fitted on the plain signal, and the offline path refuses recordings made otherwise
    // (pipeline/bci2000_recording.py). SourceFilter runs before the file writer, so it changes the
    // recording too; AlignChannels changes only what this module sees, which would break train/serve
    // parity silently.
    for (const char *p : {"NotchFilter", "HighPassFilter", "LowPassFilter", "AlignChannels"})
        if (OptionalParameter(p, 0) != 0)
            bcierr << p << " must be 0: the classifier was fitted on the plain signal";
    // Onsets are sample indices within the run; a run split over several files breaks the offline match.
    if (!std::string(OptionalParameter("FileSplittingCondition", "")).empty())
        bcierr << "FileSplittingCondition must be empty: each run must be one .dat file";
    // BCI2000 reads unitless gains as microvolts only when no channel has a unit; a mix would reach this
    // module in different units from what the offline reader assumes.
    if (Parameters->Exists("SourceChGain"))
    {
        ParamRef gains = Parameter("SourceChGain");
        const int n = std::min<int>(gains->NumValues(), Parameter("SourceCh")); // BCI2000 uses the first SourceCh
        int withUnit = 0;
        for (int i = 0; i < n; ++i)
        {
            std::string g = gains(i);
            withUnit += !g.empty() && std::isalpha(static_cast<unsigned char>(g.back()));
        }
        if (withUnit != 0 && withUnit != n)
            bcierr << "SourceChGain mixes gains with and without units";
    }

    Output = SignalProperties(1, 1, SignalType::float32);
    Output.ChannelLabels()[0] = "P300Score";
}

void RiemannP300Classifier::Initialize(const SignalProperties &Input, const SignalProperties &)
{
    LoadModel(mModel);
    mHasValidation = States->Exists("Validation");
    mHasCounter = States->Exists("Counter");
    // The oldest sample a pending window can need is onset - pad; it is scored once onset + n_times + pad
    // has arrived, which happens within the block that brings it, so one window plus one block suffices.
    mCapacity = mModel.NPadded() + Input.Elements() + 1;
    mRing.assign(mModel.nChannels, std::vector<double>(mCapacity, 0.0));
    mValid.assign(mCapacity, 0);
    bciout << "Riemann P300 classifier: model " << std::string(Parameter("RiemannModelID")).substr(0, 12)
           << ", " << mModel.nChannels << " channels, epoch " << mModel.nTimes << " + 2 x " << mModel.nPad
           << " pad samples";
}

void RiemannP300Classifier::StartRun()
{
    mCount = 0;
    mPrevCode = 0;
    mHavePrevCounter = false;
    mPending.clear();
    mResults.clear();
    mRejectedCodes.clear();
    std::fill(mValid.begin(), mValid.end(), 0);
    mScored = 0;
    mRejected.clear();
}

void RiemannP300Classifier::Process(const GenericSignal &Input, GenericSignal &Output)
{
    const StateRef codeState = State("StimulusCode"), typeState = State("StimulusType");
    const StateRef validState = OptionalState("Validation", 1), counterState = OptionalState("Counter", 0);
    for (int i = 0; i < Input.Elements(); ++i)
    {
        const long long slot = mCount % mCapacity;
        for (int c = 0; c < mModel.nChannels; ++c)
            mRing[c][slot] = Input(c, i);
        // Invalid: a sample the source filled in for a lost packet, or the first sample after a break in
        // the device's sample counter (a reset, or blocks dropped); pipeline/bci2000_recording.py
        // invalid_samples() is the same rule.
        bool valid = !mHasValidation || uint32_t(validState(i)) != 0;
        if (mHasCounter)
        {
            const uint32_t counter = counterState(i);
            if (mHavePrevCounter && counter != uint32_t(mPrevCounter + 1))
                valid = false;
            mPrevCounter = counter;
            mHavePrevCounter = true;
        }
        mValid[slot] = valid ? 1 : 0;
        int code = static_cast<int>(uint32_t(codeState(i)));
        if (code > 0 && code != mPrevCode)
            mPending.push_back({code, static_cast<int>(uint32_t(typeState(i))), mCount});
        mPrevCode = code;
        ++mCount;
    }
    Evaluate();

    State("StimulusCodeRes") = 0;
    State("StimulusTypeRes") = 0;
    State("ClassifierRejectedCode") = 0;
    StateRef scoreState = State("ClassifierScore");
    scoreState.AsFloat() = 0.0f;
    State("ClassifierOnsetSample") = 0;
    Output(0, 0) = 0;
    if (!mResults.empty())
    {
        const Result &r = mResults.front();
        State("StimulusCodeRes") = r.code;
        State("StimulusTypeRes") = r.type;
        scoreState.AsFloat() = static_cast<float>(r.score);
        State("ClassifierOnsetSample") = static_cast<State::ValueType>(r.onset);
        Output(0, 0) = r.score;
        mResults.pop_front();
    }
    if (!mRejectedCodes.empty())
    {
        State("ClassifierRejectedCode") = mRejectedCodes.front();
        mRejectedCodes.pop_front();
    }
}

void RiemannP300Classifier::Evaluate()
{
    const long long last = mCount - 1;
    while (!mPending.empty())
    {
        const Flash f = mPending.front();
        const long long start = f.onset - mModel.nPad, stop = f.onset + mModel.nTimes + mModel.nPad; // [start, stop)
        if (stop - 1 > last)
            break; // not all samples yet
        mPending.pop_front();
        const char *reason = nullptr;
        if (start < 0)
            reason = "window starts before the run";
        else if (start < mCount - mCapacity)
            reason = "window no longer buffered";
        else
        {
            for (long long s = start; s < stop && !reason; ++s)
                if (!mValid[s % mCapacity])
                    reason = "lost samples in the window";
        }
        double score = 0;
        if (!reason)
        {
            Eigen::MatrixXd seg(mModel.nChannels, mModel.NPadded());
            for (long long s = start; s < stop; ++s)
                for (int c = 0; c < mModel.nChannels; ++c)
                    seg(c, s - start) = mRing[c][s % mCapacity];
            reason = RiemannP300::Decide(mModel, seg, score);
        }
        if (reason)
        {
            ++mRejected[reason];
            mRejectedCodes.push_back(f.code);
        }
        else
        {
            ++mScored;
            mResults.push_back({f.code, f.type, f.onset, score});
        }
    }
}

void RiemannP300Classifier::StopRun()
{
    std::ostringstream oss;
    oss << "Riemann P300 classifier: " << mScored << " flash(es) scored";
    for (const auto &r : mRejected)
        oss << ", " << r.second << " rejected (" << r.first << ")";
    if (!mPending.empty())
        oss << ", " << mPending.size() << " unfinished at the end of the run";
    bciout << oss.str();
}
