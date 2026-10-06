////////////////////////////////////////////////////////////////////////////////
// Test driver for BayesianStopping.h and ThaiGPT.h, used by tests/test_bci2000_stopping.py.
//
//   stopping_cli stop < input
//     input: nChars lam theta tMin tMax a b clip / prior (nChars values) / nFlashes, then per
//            flash: groupSize, the group's indices, score, pi
//     output: per flash "t maxP argmax stop charIdx" (charIdx -1 when not stopping), then the
//             final log-posterior on one line (%.17g)
//   stopping_cli lm FILE.lmbin < contexts
//     contexts: one UTF-8 context per line (an empty line is the empty context)
//     output: per context, the prior over the file's matrix characters (%.17g), then the time per
//             prior in microseconds on a final "us <mean>" line
////////////////////////////////////////////////////////////////////////////////
#include "../BayesianStopping.h"
#include "../ThaiGPT.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

static int Stop()
{
    BayesianStopping::Controller c;
    std::cin >> c.nChars >> c.lam >> c.theta >> c.tMin >> c.tMax >> c.sigmoidA >> c.sigmoidB >> c.llrClip;
    std::vector<double> prior(c.nChars);
    for (double &p : prior)
        std::cin >> p;
    c.Reset(prior);
    int nFlashes = 0;
    std::cin >> nFlashes;
    for (int f = 0; f < nFlashes; ++f)
    {
        int n = 0;
        std::cin >> n;
        std::vector<int> group(n);
        for (int &g : group)
            std::cin >> g;
        double s = 0, pi = 0;
        std::cin >> s >> pi;
        int idx = -1;
        bool stop = c.Update(group, s, pi, idx);
        std::printf("%d %.17g %d %d %d\n", c.t, c.MaxProbability(), c.Argmax(), stop ? 1 : 0, idx);
    }
    for (double v : c.logP)
        std::printf("%.17g ", v);
    std::printf("\n");
    return std::cin ? 0 : 2;
}

static int Lm(const char *path)
{
    ThaiGPT::Model m;
    std::string err = m.Load(path);
    if (!err.empty())
    {
        std::printf("error %s\n", err.c_str());
        return 1;
    }
    std::vector<int> ids;
    for (char32_t c : m.matrix)
    {
        int id = m.Id(c);
        if (id < 0)
        {
            std::printf("error matrix character missing from the vocabulary\n");
            return 1;
        }
        ids.push_back(id);
    }
    std::string line;
    double totalUs = 0;
    int n = 0;
    while (std::getline(std::cin, line))
    {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<double> p = m.Prior(line, ids);
        totalUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        ++n;
        for (double v : p)
            std::printf("%.17g ", v);
        std::printf("\n");
    }
    std::printf("us %.1f\n", n ? totalUs / n : 0.0);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !std::strcmp(argv[1], "stop"))
        return Stop();
    if (argc >= 3 && !std::strcmp(argv[1], "lm"))
        return Lm(argv[2]);
    return 2;
}
