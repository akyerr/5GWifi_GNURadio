INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_RX_OFDM RX_OFDM)

FIND_PATH(
    RX_OFDM_INCLUDE_DIRS
    NAMES RX_OFDM/api.h
    HINTS $ENV{RX_OFDM_DIR}/include
        ${PC_RX_OFDM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    RX_OFDM_LIBRARIES
    NAMES gnuradio-RX_OFDM
    HINTS $ENV{RX_OFDM_DIR}/lib
        ${PC_RX_OFDM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(RX_OFDM DEFAULT_MSG RX_OFDM_LIBRARIES RX_OFDM_INCLUDE_DIRS)
MARK_AS_ADVANCED(RX_OFDM_LIBRARIES RX_OFDM_INCLUDE_DIRS)

