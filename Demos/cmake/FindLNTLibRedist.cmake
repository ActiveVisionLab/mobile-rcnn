find_path(LNTLibRedist_ROOT_DIR LNTLib.txt
	PATHS "${CMAKE_SOURCE_DIR}/../Library" )

find_library(LNTLibRedist_LIBRARY_DEBUG
	NAMES LNTLibRedistd
	PATHS "${LNTLibRedist_ROOT_DIR}/lib/windows-vc15" "${LNTLibRedist_ROOT_DIR}/lib/linux-cuda9.0" {CMAKE_LIB_PATH}
)

find_library(LNTLibRedist_LIBRARY_RELEASE
	NAMES LNTLibRedist
	PATHS "${LNTLibRedist_ROOT_DIR}/lib/windows-vc15" "${LNTLibRedist_ROOT_DIR}/lib/linux-cuda9.0" {CMAKE_LIB_PATH}
)

find_path(LNTLibRedist_INCLUDE LNTLibRedist.h
	PATHS "${LNTLibRedist_ROOT_DIR}/include"
)

if (LNTLibRedist_LIBRARY_DEBUG AND LNTLibRedist_INCLUDE AND LNTLibRedist_LIBRARY_RELEASE)
	set(LNTLibRedist_FOUND TRUE)
else ()
	set(LNTLibRedist_FOUND FALSE)
endif()
