# - Find GTest
# This module defines
#  GTest_INCLUDE_DIR, where to find GTest include files.
#  GTest_LIBRARY_DEBUG, where to find the GTest library.
#  GTest_LIBRARY_RELEASE, where to find the GTest library.
#  GTest_MAIN_LIBRARY_DEBUG, where to find the GTest library.
#  GTest_MAIN_LIBRARY_RELEASE, where to find the GTest library.
#  GTest_FOUND, If false, do not try to use OpenNI.

IF(MSVC_IDE)
	FIND_PATH(PNG_ROOT include/png.h HINTS "C:/all_libs/libpng")
	FIND_PATH(PNG_INCLUDE_DIR png.h HINTS "${PNG_ROOT}/include")
	FIND_LIBRARY(PNG_LIBRARY_DEBUG libpng16d HINTS "${PNG_ROOT}/lib" "${PNG_ROOT}/bin")
	FIND_LIBRARY(PNG_LIBRARY_RELEASE libpng16 HINTS "${PNG_ROOT}/lib" "${PNG_ROOT}/bin")
ELSEIF(APPLE)
	FIND_PATH(PNG_ROOT include/png.h HINTS /usr/local )
	MESSAGE(FATAL_ERROR "PNG need to be set up on this platform.")
ELSEIF("${CMAKE_SYSTEM}" MATCHES "Linux")
	MESSAGE(FATAL_ERROR "PNG need to be set up on this platform.")
ELSE()
  MESSAGE(FATAL_ERROR "PNG not currently set up to work on this platform.")
ENDIF()

IF(PNG_LIBRARY_RELEASE AND PNG_INCLUDE_DIR AND PNG_ROOT)
	SET(PNG_FOUND TRUE)
ELSE()
	SET(PNG_FOUND FALSE)
ENDIF()