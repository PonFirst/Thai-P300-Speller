////////////////////////////////////////////////////////////////////////////////
// RiemannP300Classifier: BCI2000 signal-processing filter that scores every checkerboard flash
// with the classifier fitted by pipeline/fit_from_recording.py (xDAWN covariances -> tangent
// space -> shrinkage LDA), exported by pipeline/bci2000_export.py.
//
// Train/serve parity is the design goal. The filter cuts each flash's epoch exactly as
// pipeline/epoching.py does on a BCI2000 recording: the flash onset is the first sample whose
// StimulusCode differs from the previous sample's and is nonzero; the raw window runs from
// onset - pad to onset + n_times + pad; it is band-passed with zero phase on that window alone
// (scipy's sosfiltfilt, RiemannP300Math.h), cropped and scored. Zero phase needs the samples
// after the epoch, so a flash is scored n_times + pad samples after its onset (1.3 s with the
// default 0.8 s epoch and 0.5 s pad), the same latency as the Python inference service.
//
// A window is rejected, and no result reported, when it starts before the run; contains a sample
// the source filled in for a lost Bluetooth packet (Validation = 0) or the first sample after a break
// in the device's sample counter; is not finite; fails the quality gate (a flat or railed channel,
// pipeline/quality.py, RiemannQualityGate); or has linearly dependent channels. The offline paths
// (fit_from_recording, evaluate_recording, the Python service) reject the same windows.
//
// Results are reported like P3TemporalFilter's, so StimulusTask applications work unchanged:
// StimulusCodeRes / StimulusTypeRes name the flash, the output signal (1 x 1) is the LDA
// decision score. At most one result per block; the rest wait in order. The recording also
// gets, per result: ClassifierScore (the score, float32) and ClassifierOnsetSample (the onset's
// sample index within the run), and ClassifierRejectedCode for each rejected flash, so the
// online scores can be checked against an offline recomputation of the same recording.
////////////////////////////////////////////////////////////////////////////////
#ifndef RIEMANN_P300_CLASSIFIER_H
#define RIEMANN_P300_CLASSIFIER_H

#include "GenericFilter.h"
#include "RiemannP300Math.h"

#include <deque>
#include <map>
#include <string>
#include <vector>

class RiemannP300Classifier : public GenericFilter
{
  public:
    RiemannP300Classifier();
    void Preflight(const SignalProperties &, SignalProperties &) const override;
    void Initialize(const SignalProperties &, const SignalProperties &) override;
    void StartRun() override;
    void Process(const GenericSignal &, GenericSignal &) override;
    void StopRun() override;
    bool AllowsVisualization() const override
    {
        return false;
    }

  private:
    std::string LoadModel(RiemannP300::Model &) const;
    void Evaluate();

    struct Flash
    {
        int code, type;
        long long onset;
    };
    struct Result
    {
        int code, type;
        long long onset;
        double score;
    };

    RiemannP300::Model mModel;
    int mCapacity = 0;
    std::vector<std::vector<double>> mRing; // [channel][sample % capacity]
    std::vector<unsigned char> mValid;      // [sample % capacity]
    long long mCount = 0;                   // samples seen this run
    int mPrevCode = 0;
    bool mHasValidation = false, mHasCounter = false, mHavePrevCounter = false;
    uint32_t mPrevCounter = 0;
    std::deque<Flash> mPending;
    std::deque<Result> mResults;
    std::deque<int> mRejectedCodes;
    long mScored = 0;
    std::map<std::string, long> mRejected;
};

#endif // RIEMANN_P300_CLASSIFIER_H
