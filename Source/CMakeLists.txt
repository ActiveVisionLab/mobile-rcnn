#############################################
# Top-level CMakeLists.txt for Mobile R-CNN #
#############################################

cmake_minimum_required(VERSION 2.8)

PROJECT(mobile-rcnn-devel)

###########
## Build ##
###########

#################################
# Specify the CMake module path #
#################################

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

#################################
# Add additional compiler flags #
#################################

IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
ENDIF()

IF(NOT MSVC_IDE)
	SET(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
	SET(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
	SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DNDEBUG")
ENDIF()

IF(UNIX)
	IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
		SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
	ENDIF()
ENDIF(UNIX)

#########################
# Suppress auto-linking #
#########################

IF(MSVC_IDE)
	ADD_DEFINITIONS(-DUSING_CMAKE=1)
	SET(CMAKE_CXX_FLAGS "/MP4 ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF()

######################
# Add subdirectories #
######################

ADD_SUBDIRECTORY(Apps)
ADD_SUBDIRECTORY(LNTLib)
ADD_SUBDIRECTORY(LNTLibRedist)
ADD_SUBDIRECTORY(ORUtils)
