////////////////////////////////////////////////////////////////////////////////
// RiemannP300Math: the classifier arithmetic of the RiemannP300Classifier filter, with no
// BCI2000 dependencies so it compiles and is tested on its own.
//
// It is pipeline/bci2000_export.py:reference_score() step by step, which in turn is the
// fitted pipeline of classifiers/xdawn_cov_ts_regLDA.py (XdawnCovariances -> TangentSpace ->
// shrinkage LDA) applied to an epoch cut as pipeline/epoching.py cuts it:
//   1. band-pass: scipy.signal.sosfiltfilt on the padded raw window (odd extension of padlen
//      samples, forward pass from zi * first sample, backward pass from zi * last output),
//   2. crop the n_times epoch samples out of the padded window, optional baseline,
//   3. prototypes stacked above the xDAWN-filtered epoch (pyriemann covariances_EP),
//   4. sample covariance: rows centred, Z Z^T / n_times,
//   5. tangent space: logm(R^-1/2 C R^-1/2) by symmetric eigendecomposition,
//   6. weighted upper triangle (off-diagonal x sqrt(2), row-major), LDA dot + intercept.
// Decide() adds what the Python paths that score live or replayed flashes apply around that (the
// service, evaluate_recording; fit_from_recording applies the rank check only), in the same order: a
// window that is not finite is
// rejected; then pipeline/quality.py's gate on the preprocessed epoch (skip policy: a channel whose
// peak-to-peak is below the flat or above the railed threshold, or with a flat segment counted back
// from the last sample); then pipeline/epoching.py's rank check (linearly dependent channels make the
// covariance singular). It is pipeline/bci2000_export.py:reference_decision().
// tests/test_bci2000_riemann.py checks this against the numpy reference and the fitted sklearn
// pipeline, and the decisions against reference_decision.
////////////////////////////////////////////////////////////////////////////////
#ifndef RIEMANN_P300_MATH_H
#define RIEMANN_P300_MATH_H

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace RiemannP300
{

struct Model
{
    int nChannels = 0, nTimes = 0, nPad = 0, padlen = 0;
    bool hasBaseline = false;
    int baseline0 = 0, baseline1 = 0;
    std::vector<std::array<double, 6>> sos; // b0 b1 b2 a0 a1 a2, a0 == 1
    std::vector<std::array<double, 2>> zi;
    Eigen::MatrixXd xdawn;      // (nFilters, nChannels)
    Eigen::MatrixXd prototypes; // (nProto, nTimes)
    Eigen::MatrixXd refInvSqrt; // (n, n)
    Eigen::VectorXd weights;    // n (n + 1) / 2
    double intercept = 0;
    // pipeline/quality.py gate, skip policy
    bool gate = true;
    double gateFlatUV = 1.0, gateRailedUV = 400.0;
    int gateSegment = 25;
    static constexpr double RankTolerance = 1e-10; // pipeline/epoching.py RANK_TOLERANCE
    static constexpr int MinSamplesForQuality = 5; // pipeline/quality.py MIN_SAMPLES_FOR_QUALITY

    int NPadded() const
    {
        return nTimes + 2 * nPad;
    }
    int N() const
    {
        return static_cast<int>(prototypes.rows() + xdawn.rows());
    }
    // Empty when consistent, else what is wrong.
    std::string Check() const
    {
        if (nChannels < 1 || nTimes < 2 || nPad < 0 || padlen < 0)
            return "epoch dimensions must be positive (channels " + std::to_string(nChannels) + ", epoch samples " +
                   std::to_string(nTimes) + ", pad samples " + std::to_string(nPad) + ")";
        if (sos.size() != zi.size())
            return "RiemannFilterSOS and RiemannFilterZI need the same number of sections";
        for (const auto &s : sos)
        {
            if (s[3] != 1.0)
                return "every SOS section needs a0 = 1";
            for (double v : s)
                if (!std::isfinite(v))
                    return "RiemannFilterSOS contains non-finite values";
        }
        for (const auto &z : zi)
            if (!std::isfinite(z[0]) || !std::isfinite(z[1]))
                return "RiemannFilterZI contains non-finite values";
        if (gate && !(gateFlatUV >= 0 && gateRailedUV > gateFlatUV && gateSegment >= MinSamplesForQuality))
            return "quality gate thresholds are inconsistent";
        if (!sos.empty() && padlen >= NPadded())
            return "the filter's odd-extension length must be shorter than the padded epoch";
        if (sos.empty() && padlen != 0)
            return "RiemannFilterPadlen must be 0 without a band-pass";
        if (hasBaseline && !(0 <= baseline0 && baseline0 < baseline1 && baseline1 <= nTimes))
            return "RiemannBaseline must lie within the epoch";
        if (xdawn.rows() < 1 || xdawn.cols() != nChannels)
            return "RiemannXdawnFilters must have one column per channel";
        if (prototypes.rows() < 1 || prototypes.cols() != nTimes)
            return "RiemannPrototypes must have one column per epoch sample";
        if (refInvSqrt.rows() != N() || refInvSqrt.cols() != N())
            return "RiemannRefInvSqrt must be square, prototypes + xDAWN filters on a side";
        if (weights.size() != N() * (N() + 1) / 2)
            return "RiemannLDAWeights must have n (n + 1) / 2 entries";
        return "";
    }
};

// scipy.signal.sosfilt with initial state zi (direct form II transposed, as scipy's _sosfilt),
// in place on x.
inline void SosFilt(const Model &m, std::vector<double> &x, std::vector<std::array<double, 2>> zi)
{
    const size_t nSec = m.sos.size();
    for (double &sample : x)
    {
        double cur = sample;
        for (size_t s = 0; s < nSec; ++s)
        {
            const std::array<double, 6> &c = m.sos[s];
            double y = c[0] * cur + zi[s][0];
            zi[s][0] = c[1] * cur - c[4] * y + zi[s][1];
            zi[s][1] = c[2] * cur - c[5] * y;
            cur = y;
        }
        sample = cur;
    }
}

// scipy.signal.sosfiltfilt(sos, x) with padtype "odd" and padlen m.padlen, on one channel.
inline std::vector<double> FiltFilt(const Model &m, const std::vector<double> &x)
{
    if (m.sos.empty())
        return x;
    const int n = m.padlen, len = static_cast<int>(x.size());
    std::vector<double> ext;
    ext.reserve(len + 2 * n);
    for (int k = n; k >= 1; --k) // 2 x[0] - x[n..1]
        ext.push_back(2 * x[0] - x[k]);
    ext.insert(ext.end(), x.begin(), x.end());
    for (int k = len - 2; k >= len - 1 - n; --k) // 2 x[-1] - x[-2..-(n+1)]
        ext.push_back(2 * x[len - 1] - x[k]);

    std::vector<std::array<double, 2>> zi(m.zi.size());
    for (size_t s = 0; s < zi.size(); ++s)
        zi[s] = {m.zi[s][0] * ext.front(), m.zi[s][1] * ext.front()};
    SosFilt(m, ext, zi);
    std::vector<double> rev(ext.rbegin(), ext.rend());
    for (size_t s = 0; s < zi.size(); ++s)
        zi[s] = {m.zi[s][0] * rev.front(), m.zi[s][1] * rev.front()};
    SosFilt(m, rev, zi);
    std::vector<double> out(len);
    for (int k = 0; k < len; ++k) // reverse back and drop the extensions
        out[k] = rev[rev.size() - 1 - (n + k)];
    return out;
}

// Band-pass, crop and baseline one padded raw window, segment(channel, sample) in uV of shape
// (nChannels, NPadded()): epoching.preprocess_segment. Returns (nChannels, nTimes).
inline Eigen::MatrixXd Preprocess(const Model &m, const Eigen::MatrixXd &segment)
{
    Eigen::MatrixXd x(m.nChannels, m.nTimes);
    std::vector<double> row(m.NPadded());
    for (int c = 0; c < m.nChannels; ++c)
    {
        for (int t = 0; t < m.NPadded(); ++t)
            row[t] = segment(c, t);
        std::vector<double> f = FiltFilt(m, row);
        for (int t = 0; t < m.nTimes; ++t)
            x(c, t) = f[m.nPad + t];
    }
    if (m.hasBaseline)
    {
        Eigen::VectorXd mean = x.middleCols(m.baseline0, m.baseline1 - m.baseline0).rowwise().mean();
        x.colwise() -= mean;
    }
    return x;
}

// pipeline/quality.py channel_levels + gate_epoch (skip): true when any channel is red.
inline bool FailsGate(const Model &m, const Eigen::MatrixXd &x)
{
    const int T = static_cast<int>(x.cols());
    if (T < Model::MinSamplesForQuality)
        return true; // "unknown": too short to judge
    for (int c = 0; c < x.rows(); ++c)
    {
        const double ptp = x.row(c).maxCoeff() - x.row(c).minCoeff();
        if (ptp < m.gateFlatUV || ptp > m.gateRailedUV)
            return true;
        if (T <= m.gateSegment)
            continue; // the whole window is the only segment, already checked
        const int nFull = T / m.gateSegment, first = T - nFull * m.gateSegment;
        for (int k = 0; k < nFull; ++k)
        {
            auto seg = x.row(c).segment(first + k * m.gateSegment, m.gateSegment);
            if (seg.maxCoeff() - seg.minCoeff() < m.gateFlatUV)
                return true;
        }
    }
    return false;
}

// pipeline/epoching.py is_rank_deficient: the channels are linearly dependent.
inline bool RankDeficient(const Eigen::MatrixXd &x)
{
    Eigen::MatrixXd xc = x.colwise() - x.rowwise().mean();
    Eigen::MatrixXd cov = (xc * xc.transpose()) / static_cast<double>(x.cols());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov, Eigen::EigenvaluesOnly);
    if (eig.info() != Eigen::Success)
        return true;
    const double lo = eig.eigenvalues()(0), hi = eig.eigenvalues()(eig.eigenvalues().size() - 1);
    return !(hi > 0 && lo > Model::RankTolerance * hi);
}

// The LDA decision score of a preprocessed epoch x (nChannels, nTimes). False if the tangent-space
// log is undefined (a covariance that is not positive definite).
inline bool ScorePreprocessed(const Model &m, const Eigen::MatrixXd &x, double &score)
{
    const int nProto = static_cast<int>(m.prototypes.rows()), n = m.N();
    Eigen::MatrixXd z(n, m.nTimes);
    z.topRows(nProto) = m.prototypes;
    z.bottomRows(m.xdawn.rows()) = m.xdawn * x;
    Eigen::VectorXd mean = z.rowwise().mean();
    z.colwise() -= mean;
    Eigen::MatrixXd cov = (z * z.transpose()) / static_cast<double>(m.nTimes);
    Eigen::MatrixXd s = m.refInvSqrt * cov * m.refInvSqrt;
    s = 0.5 * (s + s.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(s);
    if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <= 0)
        return false;
    Eigen::MatrixXd logS =
        eig.eigenvectors() * eig.eigenvalues().array().log().matrix().asDiagonal() * eig.eigenvectors().transpose();
    double acc = 0;
    int k = 0;
    const double root2 = std::sqrt(2.0);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j, ++k)
            acc += (i == j ? logS(i, j) : root2 * logS(i, j)) * m.weights(k);
    score = acc + m.intercept;
    return std::isfinite(score);
}

// The LDA decision score for one padded raw window, with no gate or rank check (the classifier
// arithmetic alone, as reference_score). False if the window is not finite or cannot be scored.
inline bool Score(const Model &m, const Eigen::MatrixXd &segment, double &score)
{
    if (segment.rows() != m.nChannels || segment.cols() != m.NPadded() || !segment.allFinite())
        return false;
    return ScorePreprocessed(m, Preprocess(m, segment), score);
}

// What the filter does with one padded raw window: returns nullptr and sets score, or the reason
// it is not scored ("non_finite", "channel_quality", "rank_deficient", "not_positive_definite"),
// as reference_decision().
inline const char *Decide(const Model &m, const Eigen::MatrixXd &segment, double &score)
{
    if (segment.rows() != m.nChannels || segment.cols() != m.NPadded() || !segment.allFinite())
        return "non_finite";
    Eigen::MatrixXd x = Preprocess(m, segment);
    if (m.gate && FailsGate(m, x))
        return "channel_quality";
    if (RankDeficient(x))
        return "rank_deficient";
    if (!ScorePreprocessed(m, x, score))
        return "not_positive_definite";
    return nullptr;
}

} // namespace RiemannP300

#endif // RIEMANN_P300_MATH_H
