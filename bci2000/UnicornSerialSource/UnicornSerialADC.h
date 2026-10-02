////////////////////////////////////////////////////////////////////////////////
// UnicornSerialADC: BCI2000 source filter for the g.tec Unicorn Hybrid Black on macOS.
//
// BCI2000's own gUnicornADC needs g.tec's Windows-only Unicorn.dll. This filter reads the
// headset's Bluetooth serial port directly using g.tec's published protocol, and keeps
// gUnicornADC's interface: the same parameters (SourceCh, SourceChList, SampleBlockSize,
// SamplingRate, SourceChGain, SourceChOffset, ChannelNames, TestMode) and the same nine
// per-sample states (Counter, AccelerationX/Y/Z, GyroscopeX/Y/Z, Battery, Validation), so a
// recording has the same layout either way. Differences, all deliberate:
//   - Lost Bluetooth packets are filled. BCI2000's clock is the sample count, so a lost
//     sample that took no slot would shift everything after it against the stimulus states.
//     A gap of n samples becomes n copies of the last real sample, with Validation = 0 on
//     each (Validation = 1 on every real sample). Gaps over MaxGapFill end the run.
//   - Acceleration and gyroscope states hold the raw int16 counts (read them as int32;
//     g = counts / 4096, deg/s = counts / 32.8). gUnicornADC rounds physical units into the
//     integer states, which loses almost everything.
//   - TestMode is accepted for parameter-file compatibility but must be 0: the Bluetooth
//     protocol has no test-signal command.
//   - TcpAddress reads the same bytes over TCP (tools/fake_unicorn_bt.py), whose data marks
//     itself synthetic in the gyroscope states (0x5A5A on all three axes).
////////////////////////////////////////////////////////////////////////////////
#ifndef UNICORN_SERIAL_ADC_H
#define UNICORN_SERIAL_ADC_H

#include "BufferedADC.h"
#include "ByteSource.h"
#include "UnicornProtocol.h"

#include <chrono>
#include <deque>
#include <memory>
#include <vector>

class UnicornSerialADC : public BufferedADC
{
  public:
    UnicornSerialADC();
    ~UnicornSerialADC();
    void OnPublish() override;
    void OnAutoConfig() override;
    void OnPreflight(SignalProperties &Output) const override;
    void OnInitialize(const SignalProperties &Output) override;
    void OnStartAcquisition() override;
    void DoAcquire(GenericSignal &Output) override;
    void OnStopAcquisition() override;

  private:
    std::string ResolvePort() const;
    void ReadMore();
    void Report();

    std::unique_ptr<Unicorn::ByteSource> mpSource;
    std::unique_ptr<Unicorn::PacketFramer> mpFramer;
    std::unique_ptr<Unicorn::GapFiller> mpFiller;
    std::deque<Unicorn::Sample> mPending;
    std::vector<Unicorn::Payload> mPayloads;
    std::vector<int> mChannelList;
    std::string mPort, mTcpAddress;
    long mMaxGapFill = 0;
    double mDataTimeoutS = 3.0;
    std::chrono::steady_clock::time_point mLastData;
    long mBadPayloads = 0;
    bool mSyntheticReported = false;
    bool mCounterPrecisionWarned = false;
};

#endif // UNICORN_SERIAL_ADC_H
