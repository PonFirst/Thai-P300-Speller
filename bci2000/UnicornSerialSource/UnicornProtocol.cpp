////////////////////////////////////////////////////////////////////////////////
// UnicornProtocol: see UnicornProtocol.h. Mirrors collection/unicorn_protocol.py
// line for line where it can, so the two are easy to compare.
////////////////////////////////////////////////////////////////////////////////
#include "UnicornProtocol.h"

#include <algorithm>

namespace Unicorn
{

const uint8_t StartAcquisition[3] = {0x61, 0x7C, 0x87};
const uint8_t StopAcquisition[3] = {0x63, 0x5C, 0xC5};

namespace
{
const uint8_t Header[2] = {0xC0, 0x00};
const uint8_t Footer[2] = {0x0D, 0x0A};
const uint8_t Ack[3] = {0x00, 0x00, 0x00};
const size_t BatteryAt = 2, EegAt = 3, AccAt = 27, GyrAt = 33, CounterAt = 39, FooterAt = 43;

int16_t LittleInt16(const uint8_t *p)
{
    return static_cast<int16_t>(static_cast<uint16_t>(p[0] | (p[1] << 8)));
}
} // namespace

bool Sample::IsSynthetic() const
{
    return gyrCounts[0] == SyntheticGyroCount && gyrCounts[1] == SyntheticGyroCount &&
           gyrCounts[2] == SyntheticGyroCount;
}

Sample DecodePayload(const uint8_t *p, size_t length)
{
    if (length != PayloadLength)
        throw PayloadError("payload must be 45 bytes, got " + std::to_string(length));
    if (p[0] != Header[0] || p[1] != Header[1])
        throw PayloadError("bad header");
    if (p[FooterAt] != Footer[0] || p[FooterAt + 1] != Footer[1])
        throw PayloadError("bad footer");
    Sample s;
    s.batteryPct = 100.0 * (p[BatteryAt] & 0x0F) / 15.0;
    for (int i = 0; i < NumEeg; ++i)
    {
        const uint8_t *b = p + EegAt + 3 * i;
        int32_t v = (int32_t(b[0]) << 16) | (int32_t(b[1]) << 8) | int32_t(b[2]);
        if (v & 0x800000)
            v -= 0x1000000; // sign-extend 24 bits
        s.eegUv[i] = v * EegUvPerCount;
    }
    for (int i = 0; i < 3; ++i)
    {
        s.accCounts[i] = LittleInt16(p + AccAt + 2 * i);
        s.gyrCounts[i] = LittleInt16(p + GyrAt + 2 * i);
    }
    const uint8_t *c = p + CounterAt;
    s.counter = static_cast<int32_t>(uint32_t(c[0]) | (uint32_t(c[1]) << 8) | (uint32_t(c[2]) << 16) |
                                     (uint32_t(c[3]) << 24));
    return s;
}

Payload EncodePayload(const std::array<int32_t, NumEeg> &eeg, int32_t counter, uint8_t batteryNibble,
                      const std::array<int16_t, 3> &acc, const std::array<int16_t, 3> &gyr)
{
    Payload out{};
    out[0] = Header[0];
    out[1] = Header[1];
    out[BatteryAt] = batteryNibble & 0x0F;
    for (int i = 0; i < NumEeg; ++i)
    {
        if (eeg[i] < -(1 << 23) || eeg[i] >= (1 << 23))
            throw std::out_of_range("EEG count does not fit 24 bits");
        uint32_t v = static_cast<uint32_t>(eeg[i]) & 0xFFFFFF;
        out[EegAt + 3 * i] = (v >> 16) & 0xFF;
        out[EegAt + 3 * i + 1] = (v >> 8) & 0xFF;
        out[EegAt + 3 * i + 2] = v & 0xFF;
    }
    for (int i = 0; i < 3; ++i)
    {
        uint16_t a = static_cast<uint16_t>(acc[i]), g = static_cast<uint16_t>(gyr[i]);
        out[AccAt + 2 * i] = a & 0xFF;
        out[AccAt + 2 * i + 1] = a >> 8;
        out[GyrAt + 2 * i] = g & 0xFF;
        out[GyrAt + 2 * i + 1] = g >> 8;
    }
    uint32_t c = static_cast<uint32_t>(counter);
    for (int i = 0; i < 4; ++i)
        out[CounterAt + i] = (c >> (8 * i)) & 0xFF;
    out[FooterAt] = Footer[0];
    out[FooterAt + 1] = Footer[1];
    return out;
}

PacketFramer::PacketFramer(bool expectAck) : mAwaitingAck(expectAck)
{
}

void PacketFramer::Feed(const uint8_t *data, size_t length, std::vector<Payload> &out)
{
    std::vector<uint8_t> &buf = mBuf;
    buf.insert(buf.end(), data, data + length);
    if (mAwaitingAck)
    {
        if (buf.empty())
            return;
        if (buf[0] != 0x00 || buf.size() >= sizeof(Ack))
        {
            if (buf.size() >= sizeof(Ack) && std::equal(Ack, Ack + sizeof(Ack), buf.begin()))
            {
                buf.erase(buf.begin(), buf.begin() + sizeof(Ack));
                ackReceived = true;
            }
            mAwaitingAck = false;
        }
        else
            return; // 1-2 zero bytes: wait for the rest of the ack
    }
    size_t i = 0;
    for (;;)
    {
        auto it = std::search(buf.begin() + i, buf.end(), Header, Header + sizeof(Header));
        if (it == buf.end())
        {
            size_t keep = (!buf.empty() && buf.back() == Header[0] && buf.size() > i) ? 1 : 0; // a split header
            Discard(buf.size() - keep - i);
            i = buf.size() - keep;
            break;
        }
        size_t j = it - buf.begin();
        if (j > i)
        {
            Discard(j - i);
            i = j;
        }
        if (buf.size() - j < PayloadLength)
            break; // a frame is on its way
        if (buf[j + FooterAt] == Footer[0] && buf[j + FooterAt + 1] == Footer[1])
        {
            Payload p;
            std::copy(buf.begin() + j, buf.begin() + j + PayloadLength, p.begin());
            out.push_back(p);
            ++frames;
            mInGarbage = false;
            i = j + PayloadLength;
        }
        else
        {
            Discard(1); // a header-looking pair inside data
            i = j + 1;
        }
    }
    buf.erase(buf.begin(), buf.begin() + i);
}

void PacketFramer::Discard(size_t n)
{
    if (n == 0)
        return;
    bytesDiscarded += static_cast<long>(n);
    if (!mInGarbage)
    {
        ++resyncs;
        mInGarbage = true;
    }
}

CounterTracker::Step CounterTracker::Update(int32_t counter)
{
    ++samples;
    if (!hasLast)
    {
        hasLast = true;
        last = counter;
        return {first, 0};
    }
    int64_t step = int64_t(counter) - int64_t(last);
    last = counter;
    if (step == 1)
        return {ok, 0};
    if (step == 0)
    {
        ++duplicates;
        return {duplicate, 0};
    }
    if (step < 0)
    {
        ++resets; // the device restarted its counter, or the frame was corrupt
        return {reset, 0};
    }
    long lost = static_cast<long>(step - 1);
    missing += lost;
    ++gaps;
    longestGap = std::max(longestGap, lost);
    return {gap, lost};
}

bool GapFiller::Push(const Sample &s, std::deque<Sample> &out)
{
    CounterTracker::Step step = mTracker.Update(s.counter);
    if (step.kind == CounterTracker::duplicate)
        return true;
    if (step.kind == CounterTracker::gap)
    {
        if (step.missing > mMaxFill)
        {
            refusedGap = step.missing;
            return false;
        }
        for (long k = 1; k <= step.missing; ++k)
        {
            Sample f = mLast;
            f.counter = static_cast<int32_t>(mLast.counter + k);
            f.filled = true;
            out.push_back(f);
        }
        filled += step.missing;
    }
    out.push_back(s);
    mLast = s;
    mHaveLast = true;
    return true;
}

} // namespace Unicorn
