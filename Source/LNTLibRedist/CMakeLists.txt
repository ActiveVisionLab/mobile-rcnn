##################################
# CMakeLists.txt for InputSource #
##################################

###########################
# Specify the target name #
###########################

SET(targetname LNTLibRedist)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDNN.cmake)

########################################
# Specify other params for the library #
########################################

ADD_DEFINITIONS(-DLIBRARY_EXPORTS)

IF(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
ENDIF()

#############################
# Specify the project files #
#############################

SET(LNTLIB_REDIST_SOURCES
LNTLibRedist.cpp
)

SET(LNTLIB_REDIST_HEADERS
LNTLibRedist.h
)

#################################################################
# Collect the project files into sources, headers and templates #
#################################################################

SET(sources
${LNTLIB_REDIST_SOURCES}
)

SET(headers
${LNTLIB_REDIST_HEADERS}
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${sources} ${headers})

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${LNTLIB_REDIST_SOURCES} ${LNTLIB_REDIST_HEADERS})

##########################################
# Specify the target and where to put it #
##########################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/SetGPUSharedLibTarget.cmake)

TARGET_LINK_LIBRARIES(${targetname}
	LNTLib
	ORUtils
)

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkCUDNN.cmake)