////////////////////////////////////////////////////////////////////////////////
// CheckerboardSequence: the checkerboard flash groups for one character selection.
//
// A C++ port of collection/flash_sequence.py (build_flash_sequence and
// validate_no_adjacent_co_flash), which stays the reference; the test driver in
// tests/ lets tests/test_bci2000_checkerboard.py check this against it.
//
// Characters split into two colors by flat-index parity (even: white, odd: black).
// Each repetition shuffles both colors into a virtual v_rows x v_cols matrix (6 x 6
// for 72 characters) and flashes, in order: white rows, black rows, white columns,
// black columns. Every character appears in exactly two groups per repetition.
// Flat-index parity equals true checkerboard parity only when the display grid
// has an odd number of columns (8 x 9 is safe, 9 x 8 is not): see flash_sequence.py.
////////////////////////////////////////////////////////////////////////////////
#ifndef CHECKERBOARD_SEQUENCE_H
#define CHECKERBOARD_SEQUENCE_H

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Checkerboard
{

typedef std::vector<int> Group;

// (v_rows, v_cols) as close to square as possible with v_rows <= v_cols.
inline std::pair<int, int> VirtualShape(int half)
{
    if (half <= 0)
        throw std::invalid_argument("half must be positive");
    std::pair<int, int> best(1, half);
    for (int r = 1; r <= static_cast<int>(std::sqrt(double(half))); ++r)
        if (half % r == 0)
            best = std::make_pair(r, half / r);
    return best;
}

// Shuffle uses the caller's random generator, so a BCI2000 run is reproducible from
// its RandomSeed parameter.
template <class URBG> std::vector<Group> BuildFlashSequence(int nChars, int nRepetitions, URBG &rng)
{
    if (nChars <= 0 || nChars % 2 != 0)
        throw std::invalid_argument("the number of characters must be even and positive");
    std::vector<int> white, black;
    for (int i = 0; i < nChars; ++i)
        (i % 2 == 0 ? white : black).push_back(i);
    const int half = static_cast<int>(white.size());
    const std::pair<int, int> shape = VirtualShape(half);
    const int vRows = shape.first, vCols = shape.second;

    std::vector<Group> groups;
    for (int rep = 0; rep < nRepetitions; ++rep)
    {
        std::vector<int> w = white, b = black;
        std::shuffle(w.begin(), w.end(), rng);
        std::shuffle(b.begin(), b.end(), rng);
        for (const std::vector<int> *m : {&w, &b}) // rows
            for (int r = 0; r < vRows; ++r)
                groups.push_back(Group(m->begin() + r * vCols, m->begin() + (r + 1) * vCols));
        for (const std::vector<int> *m : {&w, &b}) // columns
            for (int c = 0; c < vCols; ++c)
            {
                Group g;
                for (int r = 0; r < vRows; ++r)
                    g.push_back((*m)[r * vCols + c]);
                groups.push_back(g);
            }
    }
    return groups;
}

// Pairs of characters that are 4-connected neighbours on the nRows x nCols display grid
// (row-major) and flashed together. Empty means adjacency-safe.
inline std::vector<std::pair<int, int>> AdjacentCoFlashes(const std::vector<Group> &groups, int nRows, int nCols)
{
    std::set<std::pair<int, int>> seen;
    std::vector<std::pair<int, int>> violations;
    for (const Group &g : groups)
    {
        std::set<int> members(g.begin(), g.end());
        for (int idx : g)
        {
            int r = idx / nCols, c = idx % nCols;
            const int nb[4][2] = {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
            for (const auto &p : nb)
            {
                if (p[0] < 0 || p[0] >= nRows || p[1] < 0 || p[1] >= nCols)
                    continue;
                int other = p[0] * nCols + p[1];
                if (!members.count(other))
                    continue;
                std::pair<int, int> pair(std::min(idx, other), std::max(idx, other));
                if (seen.insert(pair).second)
                    violations.push_back(pair);
            }
        }
    }
    return violations;
}

} // namespace Checkerboard

#endif // CHECKERBOARD_SEQUENCE_H
