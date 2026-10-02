////////////////////////////////////////////////////////////////////////////////
// Test driver for CheckerboardSequence.h, used by tests/test_bci2000_checkerboard.py.
//
//   checkerboard_cli N_CHARS N_REPETITIONS SEED [N_ROWS N_COLS]
//
// Prints the flash groups as JSON ({"groups": [[...], ...]}) and, given a grid, the
// adjacent co-flash violations ({"violations": [[a, b], ...]}).
////////////////////////////////////////////////////////////////////////////////
#include "../CheckerboardSequence.h"

#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char **argv)
{
    if (argc < 4)
        return 2;
    int nChars = std::atoi(argv[1]), nReps = std::atoi(argv[2]);
    std::mt19937 rng(static_cast<unsigned>(std::atoi(argv[3])));
    std::vector<Checkerboard::Group> groups;
    try
    {
        groups = Checkerboard::BuildFlashSequence(nChars, nReps, rng);
    }
    catch (const std::exception &e)
    {
        std::printf("{\"error\": \"%s\"}\n", e.what());
        return 1;
    }
    std::printf("{\"groups\": [");
    for (size_t i = 0; i < groups.size(); ++i)
    {
        std::printf("%s[", i ? ", " : "");
        for (size_t j = 0; j < groups[i].size(); ++j)
            std::printf("%s%d", j ? ", " : "", groups[i][j]);
        std::printf("]");
    }
    std::printf("]");
    if (argc >= 6)
    {
        auto v = Checkerboard::AdjacentCoFlashes(groups, std::atoi(argv[4]), std::atoi(argv[5]));
        std::printf(", \"violations\": [");
        for (size_t i = 0; i < v.size(); ++i)
            std::printf("%s[%d, %d]", i ? ", " : "", v[i].first, v[i].second);
        std::printf("]");
    }
    std::printf("}\n");
    return 0;
}
