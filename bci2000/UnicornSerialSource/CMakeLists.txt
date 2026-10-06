###########################################################################
## UnicornSerialSource: BCI2000 source module for the Unicorn Hybrid Black
## over its Bluetooth serial port: /dev/cu.UN-* on macOS (ByteSource.cpp),
## a COM port on Windows (ByteSourceWin32.cpp). See UnicornSerialADC.h.
###########################################################################

bci2000_include(SourceFilter)
bci2000_add_signal_source_module(
  UnicornSerialSource
  UnicornSerialADC.cpp
  UnicornProtocol.cpp
  ByteSource.cpp
  ByteSourceWin32.cpp
)
if(WIN32)
  target_link_libraries(UnicornSerialSource ws2_32)
endif()
