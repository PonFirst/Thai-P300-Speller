////////////////////////////////////////////////////////////////////////////////
// Test driver for UnicornProtocol.cpp, used by tests/test_bci2000_unicorn_protocol.py.
//
//   unicorn_protocol_cli [--no-ack] [--max-fill N] CHUNK...   < bytes
//
// Feeds stdin to a PacketFramer in reads of the given sizes (cycled), decodes every
// payload, runs it through a GapFiller, and prints one JSON object per output sample,
// then a final {"stats": ...} line. Bad payloads print {"bad": "<reason>"}.
////////////////////////////////////////////////////////////////////////////////
#include "../UnicornProtocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>

using namespace Unicorn;

static void PrintSample(const Sample &s)
{
    std::printf("{\"counter\": %d, \"filled\": %d, \"battery\": %.17g, \"eeg\": [", s.counter, s.filled ? 1 : 0,
                s.batteryPct);
    for (int i = 0; i < NumEeg; ++i)
        std::printf("%s%.17g", i ? ", " : "", s.eegUv[i]);
    std::printf("], \"acc\": [%d, %d, %d], \"gyr\": [%d, %d, %d], \"synthetic\": %d}\n", s.accCounts[0],
                s.accCounts[1], s.accCounts[2], s.gyrCounts[0], s.gyrCounts[1], s.gyrCounts[2],
                s.IsSynthetic() ? 1 : 0);
}

int main(int argc, char **argv)
{
    bool expectAck = true;
    long maxFill = 2500;
    std::vector<size_t> chunks;
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--no-ack"))
            expectAck = false;
        else if (!std::strcmp(argv[i], "--max-fill") && i + 1 < argc)
            maxFill = std::atol(argv[++i]);
        else
            chunks.push_back(static_cast<size_t>(std::atol(argv[i])));
    }
    if (chunks.empty())
        chunks.push_back(4096);

    std::vector<uint8_t> input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    PacketFramer framer(expectAck);
    GapFiller filler(maxFill);
    std::vector<Payload> payloads;
    std::deque<Sample> out;
    size_t pos = 0, c = 0;
    bool refused = false;
    while (pos < input.size() && !refused)
    {
        size_t n = std::min(std::max<size_t>(chunks[c++ % chunks.size()], 1), input.size() - pos);
        payloads.clear();
        framer.Feed(input.data() + pos, n, payloads);
        pos += n;
        for (const Payload &p : payloads)
        {
            try
            {
                if (!filler.Push(DecodePayload(p.data(), p.size()), out))
                {
                    refused = true;
                    break;
                }
            }
            catch (const PayloadError &e)
            {
                std::printf("{\"bad\": \"%s\"}\n", e.what());
            }
        }
        for (; !out.empty(); out.pop_front())
            PrintSample(out.front());
    }
    const CounterTracker &t = filler.Tracker();
    std::printf("{\"stats\": {\"ack\": %d, \"frames\": %ld, \"bytes_discarded\": %ld, \"resyncs\": %ld, "
                "\"samples\": %ld, \"missing\": %ld, \"duplicates\": %ld, \"resets\": %ld, \"gaps\": %ld, "
                "\"longest_gap\": %ld, \"filled\": %ld, \"refused_gap\": %ld}}\n",
                framer.ackReceived ? 1 : 0, framer.frames, framer.bytesDiscarded, framer.resyncs, t.samples,
                t.missing, t.duplicates, t.resets, t.gaps, t.longestGap, filler.filled, filler.refusedGap);
    return 0;
}
