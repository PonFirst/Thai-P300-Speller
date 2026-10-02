////////////////////////////////////////////////////////////////////////////////
// UnicornProtocol: the Unicorn Hybrid Black Bluetooth serial protocol, pure logic.
//
// A C++ port of collection/unicorn_protocol.py, which stays the reference:
// tests/test_bci2000_unicorn_protocol.py feeds the same bytes to both and requires
// identical output. No I/O and no BCI2000 dependencies, so it compiles on its own.
//
// Source: g.tec "Unicorn Bluetooth Protocol" (UnicornBluetoothProtocol.pdf).
//   Commands: start 61 7C 87, stop 63 5C C5, each acknowledged with 00 00 00.
//   Payload, 45 bytes at 250 Hz:
//     0..1   header C0 00
//     2      battery, low nibble v: percent = 100 * v / 15
//     3..26  8 EEG channels, 24-bit BIG-endian two's complement; uV = v * 4500000 / 50331642
//     27..32 accelerometer X Y Z, int16 LITTLE-endian (g = v / 4096)
//     33..38 gyroscope X Y Z, int16 LITTLE-endian (deg/s = v / 32.8)
//     39..42 sample counter, int32 LITTLE-endian (the PDF's formula has a typo; this follows its C example)
//     43..44 footer 0D 0A
////////////////////////////////////////////////////////////////////////////////
#ifndef UNICORN_PROTOCOL_H
#define UNICORN_PROTOCOL_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace Unicorn
{

constexpr double SamplingRate = 250.0;
constexpr int NumEeg = 8;
constexpr size_t PayloadLength = 45;
constexpr double EegUvPerCount = 4500000.0 / 50331642.0;
constexpr double AccGPerCount = 1.0 / 4096.0;
constexpr double GyrDpsPerCount = 1.0 / 32.8;
// tools/fake_unicorn_bt.py writes this raw gyroscope pattern so synthetic data identifies itself.
constexpr int16_t SyntheticGyroCount = 0x5A5A;

extern const uint8_t StartAcquisition[3];
extern const uint8_t StopAcquisition[3];

typedef std::array<uint8_t, PayloadLength> Payload;

struct Sample
{
    double batteryPct = 0;
    std::array<double, NumEeg> eegUv{};
    std::array<int16_t, 3> accCounts{};
    std::array<int16_t, 3> gyrCounts{};
    int32_t counter = 0;
    bool filled = false; // true: a stand-in for a sample the radio lost (see GapFiller)

    bool IsSynthetic() const;
};

struct PayloadError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// Throws PayloadError on a bad length, header or footer.
Sample DecodePayload(const uint8_t *data, size_t length);
Payload EncodePayload(const std::array<int32_t, NumEeg> &eegCounts, int32_t counter, uint8_t batteryNibble = 0x0F,
                      const std::array<int16_t, 3> &acc = {}, const std::array<int16_t, 3> &gyr = {});

// Finds payloads in a byte stream: any chunking, garbage between frames, and the 3-byte
// start acknowledge before the first one. A header counts only if the footer sits 43 bytes
// later. Skipped bytes are counted, never silently dropped.
class PacketFramer
{
  public:
    explicit PacketFramer(bool expectAck = true);
    void Feed(const uint8_t *data, size_t length, std::vector<Payload> &out);

    bool ackReceived = false;
    long frames = 0;
    long bytesDiscarded = 0;
    long resyncs = 0;

  private:
    void Discard(size_t n);
    std::vector<uint8_t> mBuf;
    bool mAwaitingAck;
    bool mInGarbage = false;
};

// The device numbers every sample, so a Bluetooth drop is a jump in the counter.
class CounterTracker
{
  public:
    enum Kind
    {
        first,
        ok,
        gap,
        duplicate,
        reset
    };
    struct Step
    {
        Kind kind;
        long missing;
    };
    Step Update(int32_t counter);

    bool hasLast = false;
    int32_t last = 0;
    long samples = 0, missing = 0, duplicates = 0, resets = 0, gaps = 0, longestGap = 0;
};

// Turns decoded samples into the sample stream BCI2000 records. BCI2000's clock is the
// sample count, so every lost sample must still occupy its slot or everything after it
// would line up with stimulus events too early. A gap of n samples is filled with n copies
// of the last real sample, marked filled (a hold, not NaN: NaN would poison every IIR filter
// downstream for the rest of the run). Duplicates are dropped. A gap longer than maxFill is
// refused (Push returns false): at that size it is a dead link or a corrupt counter, not loss
// to paper over.
class GapFiller
{
  public:
    explicit GapFiller(long maxFill) : mMaxFill(maxFill)
    {
    }
    bool Push(const Sample &, std::deque<Sample> &out);
    const CounterTracker &Tracker() const
    {
        return mTracker;
    }
    long filled = 0;
    long refusedGap = 0; // the size of the gap that made Push return false

  private:
    long mMaxFill;
    CounterTracker mTracker;
    bool mHaveLast = false;
    Sample mLast;
};

} // namespace Unicorn

#endif // UNICORN_PROTOCOL_H
