###########################################################################
## UnicornSerialSource: BCI2000 source module for the Unicorn Hybrid Black
## over its Bluetooth serial port (macOS; POSIX). See UnicornSerialADC.h.
###########################################################################

if(NOT WIN32)
  bci2000_include(SourceFilter)
  bci2000_add_signal_source_module(
    UnicornSerialSource
    UnicornSerialADC.cpp
    UnicornProtocol.cpp
    ByteSource.cpp
  )
endif()
