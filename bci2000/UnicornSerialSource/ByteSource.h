////////////////////////////////////////////////////////////////////////////////
// ByteSource: where the headset's bytes come from.
//   SerialByteSource  the headset's Bluetooth serial port: /dev/cu.UN-<serial> on macOS, COM<n> on
//                     Windows (the paired headset's "Outgoing" port)
//   TcpByteSource     the same bytes over TCP: tools/fake_unicorn_bt.py, or Bumble's rfcomm_bridge
// Mirrors SerialSource/TcpSource in collection/unicorn_relay.py.
////////////////////////////////////////////////////////////////////////////////
#ifndef UNICORN_BYTE_SOURCE_H
#define UNICORN_BYTE_SOURCE_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace Unicorn
{

struct LinkLost : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class ByteSource
{
  public:
    virtual ~ByteSource();
    virtual std::string Describe() const = 0;
    // Waits up to timeoutMs for bytes. Returns how many were read, 0 on timeout.
    // Throws LinkLost when the link closes or fails.
    size_t Read(uint8_t *buf, size_t capacity, int timeoutMs);
    // Throws LinkLost if the bytes cannot be written within timeoutMs.
    void Write(const uint8_t *data, size_t length, int timeoutMs = 2000);
    void Close();

  protected:
#ifdef _WIN32
    // ByteSourceWin32.cpp: a serial port HANDLE, or a Winsock SOCKET (both pointer-sized)
    void *mHandle = nullptr;
    std::uintptr_t mSocket = ~std::uintptr_t(0);
    int mTimeoutMs = -1; // the read timeout currently set on the serial port
#else
    int mFd = -1;
#endif
};

class SerialByteSource : public ByteSource
{
  public:
    // Opens exclusively: a second reader on the same port would steal bytes and look like packet loss.
    explicit SerialByteSource(const std::string &path);
    std::string Describe() const override
    {
        return "serial:" + mPath;
    }

  private:
    std::string mPath;
};

class TcpByteSource : public ByteSource
{
  public:
    // address is host:port
    TcpByteSource(const std::string &address, int connectTimeoutMs = 5000);
    std::string Describe() const override
    {
        return "tcp:" + mAddress;
    }

  private:
    std::string mAddress;
};

// The candidate headset ports: /dev/cu.UN-* (macOS names a paired Unicorn's port after its serial).
// Empty on Windows: its Bluetooth COM ports carry no device name, so SerialPort must name the port.
std::vector<std::string> FindHeadsetPorts();

} // namespace Unicorn

#endif // UNICORN_BYTE_SOURCE_H
