////////////////////////////////////////////////////////////////////////////////
// ByteSource on Windows: see ByteSource.h. macOS (POSIX): ByteSource.cpp.
// Same behaviour as the POSIX version: exclusive serial access, a read that returns as soon as bytes
// arrive or after timeoutMs, writes that give up instead of hanging on a stalled Bluetooth link, and
// LinkLost on any failure. The serial port is the paired headset's "Outgoing" COM port.
////////////////////////////////////////////////////////////////////////////////
#ifdef _WIN32
#include "ByteSource.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif

namespace Unicorn
{

namespace
{
std::string LastError(const std::string &what, DWORD code)
{
    char *msg = nullptr;
    ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                     nullptr, code, 0, reinterpret_cast<char *>(&msg), 0, nullptr);
    std::string text = msg ? msg : "error " + std::to_string(code);
    if (msg)
        ::LocalFree(msg);
    while (!text.empty() && (text.back() == '\r' || text.back() == '\n' || text.back() == ' '))
        text.pop_back();
    return what + ": " + text;
}

std::string SystemError(const std::string &what)
{
    return LastError(what, ::GetLastError());
}

std::string SocketError(const std::string &what)
{
    return LastError(what, static_cast<DWORD>(::WSAGetLastError()));
}

struct Winsock
{
    bool ok;
    Winsock()
    {
        WSADATA data;
        ok = ::WSAStartup(MAKEWORD(2, 2), &data) == 0;
    }
    ~Winsock()
    {
        if (ok)
            ::WSACleanup();
    }
};

void RequireWinsock()
{
    static Winsock winsock;
    if (!winsock.ok)
        throw LinkLost("Winsock could not be started");
}

HANDLE AsHandle(void *h)
{
    return static_cast<HANDLE>(h);
}

SOCKET AsSocket(std::uintptr_t s)
{
    return static_cast<SOCKET>(s);
}

const std::uintptr_t NoSocket = ~std::uintptr_t(0);
} // namespace

ByteSource::~ByteSource()
{
    Close();
}

void ByteSource::Close()
{
    if (mHandle)
        ::CloseHandle(AsHandle(mHandle));
    mHandle = nullptr;
    if (mSocket != NoSocket)
        ::closesocket(AsSocket(mSocket));
    mSocket = NoSocket;
    mTimeoutMs = -1;
}

size_t ByteSource::Read(uint8_t *buf, size_t capacity, int timeoutMs)
{
    if (mHandle)
    {
        if (timeoutMs != mTimeoutMs)
        {
            // MAXDWORD interval and multiplier with a constant: ReadFile returns at once with whatever has
            // arrived, or waits up to the constant for the first byte (documented COMMTIMEOUTS behaviour).
            COMMTIMEOUTS t = {};
            t.ReadIntervalTimeout = MAXDWORD;
            t.ReadTotalTimeoutMultiplier = timeoutMs > 0 ? MAXDWORD : 0;
            t.ReadTotalTimeoutConstant = timeoutMs > 0 ? static_cast<DWORD>(timeoutMs) : 0;
            t.WriteTotalTimeoutConstant = 2000;
            if (!::SetCommTimeouts(AsHandle(mHandle), &t))
                throw LinkLost(SystemError("SetCommTimeouts"));
            mTimeoutMs = timeoutMs;
        }
        DWORD n = 0;
        if (!::ReadFile(AsHandle(mHandle), buf, static_cast<DWORD>(capacity), &n, nullptr))
            throw LinkLost(SystemError("read"));
        return n; // 0: nothing arrived within the timeout (a serial port has no end of stream)
    }
    if (mSocket == NoSocket)
        throw LinkLost("not open");
    WSAPOLLFD p = {AsSocket(mSocket), POLLRDNORM, 0};
    int r = ::WSAPoll(&p, 1, timeoutMs);
    if (r == SOCKET_ERROR)
        throw LinkLost(SocketError("poll"));
    if (r == 0)
        return 0;
    if (p.revents & POLLRDNORM)
    {
        int n = ::recv(AsSocket(mSocket), reinterpret_cast<char *>(buf), static_cast<int>(capacity), 0);
        if (n > 0)
            return static_cast<size_t>(n);
        if (n == 0)
            throw LinkLost("the connection was closed by the other end");
        if (::WSAGetLastError() == WSAEWOULDBLOCK)
            return 0;
        throw LinkLost(SocketError("recv"));
    }
    if (p.revents & (POLLHUP | POLLERR | POLLNVAL))
        throw LinkLost("the link hung up");
    return 0;
}

void ByteSource::Write(const uint8_t *data, size_t length, int timeoutMs)
{
    if (mHandle)
    {
        COMMTIMEOUTS t = {};
        if (!::GetCommTimeouts(AsHandle(mHandle), &t))
            throw LinkLost(SystemError("GetCommTimeouts"));
        t.WriteTotalTimeoutMultiplier = 0;
        t.WriteTotalTimeoutConstant = static_cast<DWORD>(timeoutMs);
        if (!::SetCommTimeouts(AsHandle(mHandle), &t))
            throw LinkLost(SystemError("SetCommTimeouts"));
        DWORD n = 0;
        if (!::WriteFile(AsHandle(mHandle), data, static_cast<DWORD>(length), &n, nullptr))
            throw LinkLost(SystemError("write"));
        if (n < length)
            throw LinkLost("write timed out");
        return;
    }
    if (mSocket == NoSocket)
        throw LinkLost("not open");
    size_t done = 0;
    while (done < length)
    {
        WSAPOLLFD p = {AsSocket(mSocket), POLLWRNORM, 0};
        int r = ::WSAPoll(&p, 1, timeoutMs);
        if (r <= 0)
            throw LinkLost(r == 0 ? "write timed out" : SocketError("poll"));
        int n = ::send(AsSocket(mSocket), reinterpret_cast<const char *>(data + done), static_cast<int>(length - done), 0);
        if (n == SOCKET_ERROR)
        {
            if (::WSAGetLastError() == WSAEWOULDBLOCK)
                continue;
            throw LinkLost(SocketError("send"));
        }
        done += static_cast<size_t>(n);
    }
}

SerialByteSource::SerialByteSource(const std::string &path) : mPath(path)
{
    // COM10 and above only open through the \\.\ device namespace; it works for COM1-9 too.
    std::string device = path.rfind("\\\\.\\", 0) == 0 ? path : "\\\\.\\" + path;
    // Share mode 0: exclusive, as TIOCEXCL on macOS. A second reader would steal bytes.
    HANDLE h = ::CreateFileA(device.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE)
    {
        DWORD code = ::GetLastError();
        if (code == ERROR_ACCESS_DENIED)
            throw LinkLost("cannot open " + path + ": it is in use by another program (another BCI2000, the "
                                                   "relay, or Unicorn Suite)");
        throw LinkLost(LastError("cannot open " + path, code));
    }
    mHandle = h;
    DCB dcb = {};
    dcb.DCBlength = sizeof(dcb);
    if (::GetCommState(h, &dcb))
    {
        dcb.BaudRate = CBR_115200; // ignored by a Bluetooth serial port, required by a wired one
        dcb.ByteSize = 8;
        dcb.Parity = NOPARITY;
        dcb.StopBits = ONESTOPBIT;
        dcb.fBinary = TRUE;
        dcb.fParity = FALSE;
        dcb.fOutxCtsFlow = FALSE;
        dcb.fOutxDsrFlow = FALSE;
        dcb.fDsrSensitivity = FALSE;
        dcb.fOutX = FALSE;
        dcb.fInX = FALSE;
        dcb.fNull = FALSE;
        dcb.fAbortOnError = FALSE;
        dcb.fDtrControl = DTR_CONTROL_ENABLE;
        dcb.fRtsControl = RTS_CONTROL_ENABLE;
        ::SetCommState(h, &dcb);
    }
    ::SetupComm(h, 1 << 16, 1 << 12);
    ::PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR); // drop anything left over from a previous session
}

TcpByteSource::TcpByteSource(const std::string &address, int connectTimeoutMs) : mAddress(address)
{
    RequireWinsock();
    size_t colon = address.rfind(':');
    if (colon == std::string::npos)
        throw LinkLost("TCP address must be host:port, got \"" + address + "\"");
    std::string host = address.substr(0, colon), port = address.substr(colon + 1);
    addrinfo hints = {}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (::getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
        throw LinkLost("cannot resolve " + address);
    for (addrinfo *a = res; a && mSocket == NoSocket; a = a->ai_next)
    {
        SOCKET s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == INVALID_SOCKET)
            continue;
        u_long nonblocking = 1;
        ::ioctlsocket(s, FIONBIO, &nonblocking);
        int r = ::connect(s, a->ai_addr, static_cast<int>(a->ai_addrlen));
        if (r == SOCKET_ERROR && ::WSAGetLastError() == WSAEWOULDBLOCK)
        {
            WSAPOLLFD p = {s, POLLWRNORM, 0};
            int soerr = 0, len = sizeof(soerr);
            if (::WSAPoll(&p, 1, connectTimeoutMs) == 1 && (p.revents & POLLWRNORM) &&
                ::getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char *>(&soerr), &len) == 0 && soerr == 0)
                r = 0;
        }
        if (r == 0)
        {
            BOOL one = TRUE;
            ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof(one));
            mSocket = static_cast<std::uintptr_t>(s);
        }
        else
            ::closesocket(s);
    }
    ::freeaddrinfo(res);
    if (mSocket == NoSocket)
        throw LinkLost("cannot connect to " + address);
}

std::vector<std::string> FindHeadsetPorts()
{
    return {};
}

} // namespace Unicorn
#endif // _WIN32
