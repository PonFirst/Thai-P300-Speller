////////////////////////////////////////////////////////////////////////////////
// BayesianStopping: the fixed-threshold Bayesian dynamic-stopping controller, in C++ for
// BCI2000's CheckerboardSpeller.
//
// A line-by-line port of stopping/bayesian_stopping.py (log_normalize, and
// BayesianStoppingController.reset / llr / update, which the project owner implemented by hand).
// The Python stays the reference: tests/test_bci2000_stopping.py drives both with the same priors
// and flash sequences and requires the same posteriors, stop flashes and choices.
//   reset(prior):  p0 = lam * prior + (1 - lam) / n_chars; log_p = log(p0); t = 0
//   llr(s, pi):    clip((a * s + b) + log((1 - pi) / pi), -clip, +clip)
//   update(g, s, pi): log_p[g] += llr(s, pi); log_p = log_normalize(log_p); t += 1;
//                  argmax wins and stops if (t >= t_min and max p >= theta) or t >= t_max
////////////////////////////////////////////////////////////////////////////////
#ifndef BAYESIAN_STOPPING_H
#define BAYESIAN_STOPPING_H

#include <algorithm>
#include <cmath>
#include <vector>

namespace BayesianStopping
{

// stopping/bayesian_stopping.py log_normalize: log-sum-exp normalisation.
inline void LogNormalize(std::vector<double> &logP)
{
    const double m = *std::max_element(logP.begin(), logP.end());
    double s = 0;
    for (double v : logP)
        s += std::exp(v - m);
    const double logSum = m + std::log(s);
    for (double &v : logP)
        v -= logSum;
}

class Controller
{
  public:
    int nChars = 72;
    double lam = 0.7, theta = 0.90, sigmoidA = 1.0, sigmoidB = 0.0, llrClip = 8.0;
    int tMin = 6, tMax = 170;

    std::vector<double> logP;
    int t = 0;

    void Reset(const std::vector<double> &lmPrior)
    {
        logP.resize(nChars);
        for (int i = 0; i < nChars; ++i)
            logP[i] = std::log(lam * lmPrior[i] + (1 - lam) / nChars);
        t = 0;
    }

    double Llr(double s, double pi) const
    {
        const double ell = (sigmoidA * s + sigmoidB) + std::log((1 - pi) / pi);
        return std::min(std::max(ell, -llrClip), llrClip);
    }

    // Returns true (and sets charIdx) when the stopping condition is met.
    bool Update(const std::vector<int> &group, double s, double pi, int &charIdx)
    {
        const double ell = Llr(s, pi);
        for (int g : group)
            logP[g] += ell;
        LogNormalize(logP);
        ++t;
        const int maxIdx = Argmax();
        const double maxP = std::exp(logP[maxIdx]);
        if ((t >= tMin && maxP >= theta) || t >= tMax)
        {
            charIdx = maxIdx;
            return true;
        }
        return false;
    }

    // np.argmax: the first index of the maximum.
    int Argmax() const
    {
        return static_cast<int>(std::max_element(logP.begin(), logP.end()) - logP.begin());
    }
    double MaxProbability() const
    {
        return std::exp(logP[Argmax()]);
    }
};

} // namespace BayesianStopping

#endif // BAYESIAN_STOPPING_H
