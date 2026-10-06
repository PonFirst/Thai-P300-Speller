////////////////////////////////////////////////////////////////////////////////
// ByteSource on macOS (POSIX): see ByteSource.h. Windows: ByteSourceWin32.cpp.
////////////////////////////////////////////////////////////////////////////////
#ifndef _WIN32
#include "ByteSource.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <glob.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

namespace Unicorn
{

namespace
{
std::string Errno(const std::string &what)
{
    return what + ": " + std::strerror(errno);
}
} // namespace

ByteSource::~ByteSource()
{
    Close();
}

void ByteSource::Close()
{
    if (mFd >= 0)
        ::close(mFd);
    mFd = -1;
}

size_t ByteSource::Read(uint8_t *buf, size_t capacity, int timeoutMs)
{
    if (mFd < 0)
        throw LinkLost("not open");
    pollfd p = {mFd, POLLIN, 0};
    int r = ::poll(&p, 1, timeoutMs);
    if (r < 0)
    {
        if (errno == EINTR)
            return 0;
        throw LinkLost(Errno("poll"));
    }
    if (r == 0)
        return 0;
    if (p.revents & POLLIN)
    {
        ssize_t n = ::read(mFd, buf, capacity);
        if (n > 0)
            return static_cast<size_t>(n);
        if (n == 0)
            throw LinkLost("the connection was closed by the other end");
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
            return 0;
        throw LinkLost(Errno("read"));
    }
    if (p.revents & (POLLHUP | POLLERR | POLLNVAL))
        throw LinkLost("the link hung up");
    return 0;
}

void ByteSource::Write(const uint8_t *data, size_t length, int timeoutMs)
{
    // No tcdrain(): it has no timeout and hangs on a stalled Bluetooth link.
    size_t done = 0;
    while (done < length)
    {
        pollfd p = {mFd, POLLOUT, 0};
        int r = ::poll(&p, 1, timeoutMs);
        if (r <= 0)
            throw LinkLost(r == 0 ? "write timed out" : Errno("poll"));
        ssize_t n = ::write(mFd, data + done, length - done);
        if (n < 0)
        {
            if (errno == EAGAIN || errno == EINTR)
                continue;
            throw LinkLost(Errno("write"));
        }
        done += static_cast<size_t>(n);
    }
}

SerialByteSource::SerialByteSource(const std::string &path) : mPath(path)
{
    mFd = ::open(path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (mFd < 0)
        throw LinkLost(Errno("cannot open " + path));
    if (::ioctl(mFd, TIOCEXCL) != 0)
    {
        std::string err = Errno("cannot get exclusive access to " + path);
        Close();
        throw LinkLost(err);
    }
    termios t;
    if (::tcgetattr(mFd, &t) == 0)
    {
        ::cfmakeraw(&t);
        ::cfsetspeed(&t, B115200); // ignored by a Bluetooth serial port, required by a wired one
        t.c_cflag |= CLOCAL | CREAD;
        t.c_cc[VMIN] = 0;
        t.c_cc[VTIME] = 0;
        ::tcsetattr(mFd, TCSANOW, &t);
    }
    ::tcflush(mFd, TCIOFLUSH); // drop anything left over from a previous session
}

TcpByteSource::TcpByteSource(const std::string &address, int connectTimeoutMs) : mAddress(address)
{
    size_t colon = address.rfind(':');
    if (colon == std::string::npos)
        throw LinkLost("TCP address must be host:port, got \"" + address + "\"");
    std::string host = address.substr(0, colon), port = address.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (::getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0 || !res)
        throw LinkLost("cannot resolve " + address);
    std::string err = "cannot connect to " + address;
    for (addrinfo *a = res; a && mFd < 0; a = a->ai_next)
    {
        int fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0)
            continue;
        ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL) | O_NONBLOCK);
        int r = ::connect(fd, a->ai_addr, a->ai_addrlen);
        if (r != 0 && errno == EINPROGRESS)
        {
            pollfd p = {fd, POLLOUT, 0};
            int soerr = 0;
            socklen_t len = sizeof(soerr);
            if (::poll(&p, 1, connectTimeoutMs) == 1 && ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len) == 0 &&
                soerr == 0)
                r = 0;
        }
        if (r == 0)
        {
            int one = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            mFd = fd;
        }
        else
            ::close(fd);
    }
    ::freeaddrinfo(res);
    if (mFd < 0)
        throw LinkLost(err);
}

std::vector<std::string> FindHeadsetPorts()
{
    std::vector<std::string> out;
    glob_t g;
    if (::glob("/dev/cu.UN-*", 0, nullptr, &g) == 0)
        for (size_t i = 0; i < g.gl_pathc; ++i)
            out.push_back(g.gl_pathv[i]);
    ::globfree(&g);
    return out;
}

} // namespace Unicorn
#endif // _WIN32
