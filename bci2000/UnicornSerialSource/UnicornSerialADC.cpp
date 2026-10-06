////////////////////////////////////////////////////////////////////////////////
// UnicornSerialADC: see UnicornSerialADC.h.
////////////////////////////////////////////////////////////////////////////////
#include "UnicornSerialADC.h"

#include "BCIStream.h"

#include <cmath>
#include <sstream>
#ifndef _WIN32
#include <unistd.h>
#endif

RegisterFilter(UnicornSerialADC, 1); // ADC filters must be registered at location "1".

namespace
{
const int NumStateChannels = 9;
// The Unicorn Hybrid Black's factory cap, verified against g.tec's user manual (project
// CLAUDE.md). gUnicornADC's built-in defaults name a different layout; override ChannelNames
// in the parameter file if your cap differs.
const char *const DefaultChannelNames[Unicorn::NumEeg] = {"Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"};
const int ReadTimeoutMs = 100;
} // namespace

UnicornSerialADC::UnicornSerialADC()
{
}

UnicornSerialADC::~UnicornSerialADC()
{
}

void UnicornSerialADC::OnPublish()
{
    BEGIN_PARAMETER_DEFINITIONS
        "Source:Signal%20Properties int SourceCh= auto "
            "auto 1 % // number of digitized and stored channels",
        "Source:Signal%20Properties list SourceChList= 1 auto "
            " // indices of digitized channels, 0-7 (must correspond to SourceCh)",
        "Source:Signal%20Properties int SampleBlockSize= auto "
            "auto 1 % // number of samples transmitted at a time",
        "Source:Signal%20Properties float SamplingRate= 250 "
            "250 0.0 % // sample rate (noedit)",
        "Source:Signal%20Properties list SourceChGain= 1 auto "
            " // physical units per raw A/D unit",
        "Source:Signal%20Properties list SourceChOffset= 1 auto "
            " // raw A/D offset to subtract, typically 0",
        "Source:Signal%20Properties list ChannelNames= 1 auto "
            " // names of amplifier channels",
        "Source:Signal%20Properties int TestMode= 0 0 0 1 "
            " // must be 0: the Bluetooth protocol has no test signal (boolean)",
        "Source:Unicorn string SerialPort= auto auto % % "
            " // headset serial port; auto: the single /dev/cu.UN-* device (macOS). Windows: the COM port, e.g. COM5",
        "Source:Unicorn string TcpAddress= % % % % "
            " // host:port to read the same bytes over TCP instead (fake headset); empty: use SerialPort",
        "Source:Unicorn int MaxGapFill= 250 250 0 % "
            " // longest run of lost samples to fill; a longer gap ends the run",
        "Source:Unicorn float DataTimeout= 3 3 0.1 % "
            " // seconds without data before the link counts as lost",
        "Source:Unicorn float StartupTimeout= 10 10 1 % "
            " // seconds to wait for the first data after opening the port (measured: 3.5-4 s on macOS)",
    END_PARAMETER_DEFINITIONS

    BEGIN_STREAM_DEFINITIONS
        "Counter 32 0",
        "AccelerationX 32 0",
        "AccelerationY 32 0",
        "AccelerationZ 32 0",
        "GyroscopeX 32 0",
        "GyroscopeY 32 0",
        "GyroscopeZ 32 0",
        "Battery 32 0",
        "Validation 32 0",
    END_STREAM_DEFINITIONS
}

void UnicornSerialADC::OnAutoConfig()
{
    Parameter("SamplingRate") = Unicorn::SamplingRate;
    Parameter("SampleBlockSize") = 10;
    Parameter("SourceCh") = Unicorn::NumEeg;
    int n = ActualParameter("SourceCh");
    Parameter("SourceChList")->SetNumValues(n);
    Parameter("SourceChGain")->SetNumValues(n);
    Parameter("SourceChOffset")->SetNumValues(n);
    Parameter("ChannelNames")->SetNumValues(n);
    for (int i = 0; i < n; ++i)
    {
        Parameter("SourceChList")(i) = i;
        Parameter("SourceChGain")(i) = "1muV";
        Parameter("SourceChOffset")(i) = 0;
    }
    for (int i = 0; i < n && i < ActualParameter("SourceChList")->NumValues(); ++i)
    {
        int ch = ActualParameter("SourceChList")(i);
        if (ch >= 0 && ch < Unicorn::NumEeg)
            Parameter("ChannelNames")(i) = DefaultChannelNames[ch];
    }
}

std::string UnicornSerialADC::ResolvePort() const
{
    std::string port = Parameter("SerialPort");
    if (port != "auto" && !port.empty())
        return port;
#ifdef _WIN32
    return "!Set SerialPort to the headset's COM port, e.g. COM5: Windows Settings > Bluetooth & devices > "
           "More Bluetooth settings > COM Ports, the \"Outgoing\" port of UN-...";
#endif
    std::vector<std::string> ports = Unicorn::FindHeadsetPorts();
    if (ports.size() == 1)
        return ports[0];
    std::ostringstream oss;
    if (ports.empty())
        oss << "No /dev/cu.UN-* port: pair the headset in System Settings > Bluetooth and switch it on";
    else
    {
        oss << "Several headset ports, set SerialPort to one of:";
        for (const auto &p : ports)
            oss << " " << p;
    }
    return std::string("!") + oss.str();
}

void UnicornSerialADC::OnPreflight(SignalProperties &Output) const
{
    State("Counter");
    State("AccelerationX");
    State("AccelerationY");
    State("AccelerationZ");
    State("GyroscopeX");
    State("GyroscopeY");
    State("GyroscopeZ");
    State("Battery");
    State("Validation");

    if (Parameter("SamplingRate").InHertz() != Unicorn::SamplingRate)
        bcierr << "The Unicorn samples at 250 Hz only, not " << Parameter("SamplingRate");
    if (Parameter("TestMode") != 0)
        bcierr << "TestMode is not available over the Bluetooth protocol; set it to 0 "
               << "(use TcpAddress with tools/fake_unicorn_bt.py for a stand-in headset)";
    int n = Parameter("SourceCh");
    if (n < 1 || n > Unicorn::NumEeg)
        bcierr << "SourceCh must be 1-" << Unicorn::NumEeg;
    if (Parameter("SourceChList")->NumValues() != n)
        bcierr << "Number of items in SourceChList must be equal to SourceCh";
    int prev = -1;
    for (int i = 0; i < Parameter("SourceChList")->NumValues(); ++i)
    {
        int ch = Parameter("SourceChList")(i);
        if (ch < 0 || ch >= Unicorn::NumEeg)
        {
            bcierr << "SourceChList: only channels 0-7 exist";
            break;
        }
        if (ch <= prev)
        {
            bcierr << "SourceChList: channels must be unique and in ascending order";
            break;
        }
        prev = ch;
    }
    if (Parameter("ChannelNames")->NumValues() != n)
        bcierr << "A channel name must be defined for each channel";
    if (Parameter("SourceChGain")->NumValues() != n)
        bcierr << "A channel gain must be defined for each channel";
    if (Parameter("SourceChOffset")->NumValues() != n)
        bcierr << "A channel offset must be defined for each channel";
    if (Parameter("MaxGapFill") < 0)
        bcierr << "MaxGapFill must not be negative";
    if (Parameter("DataTimeout") <= 0 || Parameter("StartupTimeout") < Parameter("DataTimeout"))
        bcierr << "DataTimeout must be positive and StartupTimeout at least DataTimeout";

    std::string tcp = Parameter("TcpAddress");
    if (tcp.empty())
    {
        std::string port = ResolvePort();
        if (!port.empty() && port[0] == '!')
            bcierr << port.substr(1);
#ifdef _WIN32
        else if (port.find("COM") == std::string::npos)
            bcierr << "SerialPort must be a COM port such as COM5, not \"" << port << "\"";
#else
        else if (::access(port.c_str(), R_OK | W_OK) != 0)
            bcierr << "Cannot access headset port " << port;
#endif
    }

    Output = SignalProperties(n + NumStateChannels, Parameter("SampleBlockSize"), SignalType::float32);
    for (int i = 0; i < n && i < Parameter("ChannelNames")->NumValues(); ++i)
        Output.ChannelLabels()[i] = Parameter("ChannelNames")(i).ToString();
    const char *stateLabels[NumStateChannels] = {"@AccelerationX", "@AccelerationY", "@AccelerationZ",
                                                 "@GyroscopeX",    "@GyroscopeY",    "@GyroscopeZ",
                                                 "@Battery",       "@Counter",       "@Validation"};
    for (int i = 0; i < NumStateChannels; ++i)
        Output.ChannelLabels()[n + i] = stateLabels[i];
}

void UnicornSerialADC::OnInitialize(const SignalProperties &)
{
    mChannelList.clear();
    for (int i = 0; i < Parameter("SourceChList")->NumValues(); ++i)
        mChannelList.push_back(Parameter("SourceChList")(i));
    mTcpAddress = std::string(Parameter("TcpAddress"));
    mPort = mTcpAddress.empty() ? ResolvePort() : "";
    mMaxGapFill = Parameter("MaxGapFill");
    mDataTimeoutS = Parameter("DataTimeout");
    mStartupTimeoutS = Parameter("StartupTimeout");
    // Never write a parameter here (gUnicornADC sets ID_Amp at this point): the change is sent on to
    // the other modules, and if it reaches one that has already started running, BCI2000 reports
    // "Unexpected Param message" and shuts the system down. The device is named in the log instead
    // (OnStartAcquisition), and synthetic data marks itself in the gyroscope states.
    // The headset's first bytes arrive 3.5-4 s after the port opens (Bluetooth link setup, measured on
    // the real headset); BCI2000's own wait for the first block must outlast our startup allowance.
    SetInitialAcquisitionTimeout(Time::Seconds(mStartupTimeoutS + 2));
}

void UnicornSerialADC::OnStartAcquisition()
{
    mPending.clear();
    mBadPayloads = 0;
    mSyntheticReported = false;
    mCounterPrecisionWarned = false;
    mpFramer.reset(new Unicorn::PacketFramer(true));
    mpFiller.reset(new Unicorn::GapFiller(mMaxGapFill));
    try
    {
        if (mTcpAddress.empty())
            mpSource.reset(new Unicorn::SerialByteSource(mPort));
        else
            mpSource.reset(new Unicorn::TcpByteSource(mTcpAddress));
        mpSource->Write(Unicorn::StartAcquisition, sizeof(Unicorn::StartAcquisition));
    }
    catch (const Unicorn::LinkLost &e)
    {
        mpSource.reset();
        throw bcierr << "Could not start the headset: " << e.what();
    }
    mLastData = std::chrono::steady_clock::now();
    mGotData = false;
    bciout << "Unicorn: acquiring from " << mpSource->Describe();
}

void UnicornSerialADC::ReadMore()
{
    uint8_t buf[4096];
    size_t n = 0;
    try
    {
        n = mpSource->Read(buf, sizeof(buf), ReadTimeoutMs);
    }
    catch (const Unicorn::LinkLost &e)
    {
        throw bcierr << "Unicorn: the Bluetooth link was lost (" << e.what() << ")";
    }
    auto now = std::chrono::steady_clock::now();
    if (n == 0)
    {
        double silent = std::chrono::duration<double>(now - mLastData).count();
        if (!mGotData && silent > mStartupTimeoutS)
            throw bcierr << "Unicorn: the headset sent nothing for " << silent << " s after the port opened; "
                         << "is it switched on, paired, and not connected to another program?";
        if (mGotData && silent > mDataTimeoutS)
            throw bcierr << "Unicorn: no data from the headset for " << silent << " s; the link is down";
        return;
    }
    mLastData = now;
    mGotData = true;
    mPayloads.clear();
    mpFramer->Feed(buf, n, mPayloads);
    for (const Unicorn::Payload &p : mPayloads)
    {
        Unicorn::Sample s;
        try
        {
            s = Unicorn::DecodePayload(p.data(), p.size());
        }
        catch (const Unicorn::PayloadError &)
        {
            ++mBadPayloads;
            continue;
        }
        if (!mSyntheticReported && s.IsSynthetic())
        {
            mSyntheticReported = true;
            bciwarn << "Unicorn: SYNTHETIC DATA. The headset stand-in marked its packets "
                    << "(gyroscope states read 0x5A5A); this is not a recording of a brain.";
        }
        long filledBefore = mpFiller->filled, resetsBefore = mpFiller->Tracker().resets;
        if (!mpFiller->Push(s, mPending))
            throw bcierr << "Unicorn: " << mpFiller->refusedGap << " samples ("
                         << mpFiller->refusedGap * 1000 / Unicorn::SamplingRate << " ms) lost in one gap, more than "
                         << "MaxGapFill = " << mMaxGapFill << "; the run is stopped rather than misaligned";
        long added = mpFiller->filled - filledBefore;
        if (added > 0)
            bciwarn << "Unicorn: " << added << " sample(s) lost before counter " << s.counter << " ("
                    << added * 1000 / Unicorn::SamplingRate << " ms), filled and marked Validation = 0";
        if (mpFiller->Tracker().resets > resetsBefore)
            bciwarn << "Unicorn: the sample counter went backwards to " << s.counter
                    << " (device restart or corrupt frame); alignment after this point is not guaranteed";
    }
}

void UnicornSerialADC::DoAcquire(GenericSignal &Output)
{
    const int n = static_cast<int>(mChannelList.size());
    for (int el = 0; el < Output.Elements(); ++el)
    {
        while (mPending.empty())
            ReadMore();
        const Unicorn::Sample &s = mPending.front();
        for (int i = 0; i < n; ++i)
            Output(i, el) = static_cast<float>(s.eegUv[mChannelList[i]]);
        for (int k = 0; k < 3; ++k)
        {
            Output(n + k, el) = s.accCounts[k];
            Output(n + 3 + k, el) = s.gyrCounts[k];
        }
        Output(n + 6, el) = static_cast<float>(std::round(s.batteryPct));
        // Counter is an int32 sample index; it travels as float32 (BufferedADC copies "@" channels into
        // states), exact up to 2^24 samples (18.6 hours of acquisition). Past that, consecutive counters
        // round together and every sample looks like a counter break to the classifier.
        if (!mCounterPrecisionWarned && std::abs(static_cast<double>(s.counter)) >= 16777216.0 - 250 * 600)
        {
            mCounterPrecisionWarned = true;
            bciwarn << "Unicorn: the device sample counter is near 2^24, where it stops being exact in BCI2000's "
                    << "state; restart acquisition (Set config) within 10 minutes";
        }
        Output(n + 7, el) = static_cast<float>(s.counter);
        Output(n + 8, el) = s.filled ? 0 : 1;
        mPending.pop_front();
    }
}

void UnicornSerialADC::Report()
{
    if (!mpFiller || !mpFramer)
        return;
    const Unicorn::CounterTracker &t = mpFiller->Tracker();
    double total = double(t.samples + t.missing);
    bciout << "Unicorn: " << t.samples << " packets received, " << t.missing << " samples lost ("
           << (total > 0 ? 100.0 * t.missing / total : 0.0) << " %) in " << t.gaps << " gap(s), longest "
           << t.longestGap * 1000 / Unicorn::SamplingRate << " ms; " << mpFiller->filled << " filled; " << t.duplicates
           << " duplicate(s), " << t.resets << " counter reset(s), " << mpFramer->bytesDiscarded
           << " byte(s) discarded, " << mBadPayloads << " bad payload(s)";
}

void UnicornSerialADC::OnStopAcquisition()
{
    if (mpSource)
    {
        try
        {
            mpSource->Write(Unicorn::StopAcquisition, sizeof(Unicorn::StopAcquisition));
        }
        catch (const Unicorn::LinkLost &)
        {
        }
        mpSource->Close();
    }
    Report();
    mpSource.reset();
}
