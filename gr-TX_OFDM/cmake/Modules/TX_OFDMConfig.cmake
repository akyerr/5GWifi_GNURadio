INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_TX_OFDM TX_OFDM)

FIND_PATH(
    TX_OFDM_INCLUDE_DIRS
    NAMES TX_OFDM/api.h
    HINTS $ENV{TX_OFDM_DIR}/include
        ${PC_TX_OFDM_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    TX_OFDM_LIBRARIES
    NAMES gnuradio-TX_OFDM
    HINTS $ENV{TX_OFDM_DIR}/lib
        ${PC_TX_OFDM_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TX_OFDM DEFAULT_MSG TX_OFDM_LIBRARIES TX_OFDM_INCLUDE_DIRS)
MARK_AS_ADVANCED(TX_OFDM_LIBRARIES TX_OFDM_INCLUDE_DIRS)

