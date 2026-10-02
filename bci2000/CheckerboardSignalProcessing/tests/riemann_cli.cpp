////////////////////////////////////////////////////////////////////////////////
// Test driver for RiemannP300Math.h, used by tests/test_bci2000_riemann.py.
//
//   riemann_cli < input.txt
//
// Input, whitespace-separated numbers:
//   nChannels nTimes nPad padlen nSections nFilters nProto hasBaseline baseline0 baseline1
//   sos (nSections x 6) zi (nSections x 2) xdawn (nFilters x nChannels) prototypes (nProto x nTimes)
//   refInvSqrt (n x n) weights (n (n + 1) / 2) intercept gate gateFlatUV gateRailedUV gateSegment
//   nSegments, then each segment as NPadded rows of nChannels values (time-major, like a recording)
// Output: one line per segment, "<Score()> <Decide()>": Score() is the score (%.17g) or "reject";
// Decide() is the score or the rejection reason.
////////////////////////////////////////////////////////////////////////////////
#include "../RiemannP300Math.h"

#include <cstdio>
#include <iostream>

using namespace RiemannP300;

static void ReadMatrix(Eigen::MatrixXd &m, int rows, int cols)
{
    m.resize(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            std::cin >> m(r, c);
}

int main()
{
    Model m;
    int nSec = 0, nFilters = 0, nProto = 0, hasBaseline = 0;
    std::cin >> m.nChannels >> m.nTimes >> m.nPad >> m.padlen >> nSec >> nFilters >> nProto >> hasBaseline >>
        m.baseline0 >> m.baseline1;
    m.hasBaseline = hasBaseline != 0;
    m.sos.resize(nSec);
    m.zi.resize(nSec);
    for (auto &s : m.sos)
        for (double &v : s)
            std::cin >> v;
    for (auto &z : m.zi)
        for (double &v : z)
            std::cin >> v;
    ReadMatrix(m.xdawn, nFilters, m.nChannels);
    ReadMatrix(m.prototypes, nProto, m.nTimes);
    const int n = nFilters + nProto;
    ReadMatrix(m.refInvSqrt, n, n);
    m.weights.resize(n * (n + 1) / 2);
    for (int k = 0; k < m.weights.size(); ++k)
        std::cin >> m.weights(k);
    int gate = 0;
    std::cin >> m.intercept >> gate >> m.gateFlatUV >> m.gateRailedUV >> m.gateSegment;
    m.gate = gate != 0;
    if (!std::cin)
        return 2;
    std::string err = m.Check();
    if (!err.empty())
    {
        std::printf("error %s\n", err.c_str());
        return 1;
    }
    int nSeg = 0;
    std::cin >> nSeg;
    for (int i = 0; i < nSeg; ++i)
    {
        Eigen::MatrixXd seg(m.nChannels, m.NPadded());
        for (int t = 0; t < m.NPadded(); ++t)
            for (int c = 0; c < m.nChannels; ++c)
            {
                std::string tok;
                std::cin >> tok;
                seg(c, t) = (tok == "nan") ? std::nan("") : std::stod(tok);
            }
        double score = 0;
        if (Score(m, seg, score))
            std::printf("%.17g ", score);
        else
            std::printf("reject ");
        double decided = 0;
        const char *reason = Decide(m, seg, decided);
        if (reason)
            std::printf("%s\n", reason);
        else
            std::printf("%.17g\n", decided);
    }
    return 0;
}
