############################################
# CMakeLists.txt for Apps/MobileRCNNTester #
############################################

###########################
# Specify the target name #
###########################

SET(targetname MobileRCNNTester)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDNN.cmake)

#############################
# Specify the project files #
#############################

SET(sources
MobileRCNNTester.cpp
)

SET(headers
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${sources} ${headers})

##########################################
# Specify the target and where to put it #
##########################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/SetGPUAppTarget.cmake)

#################################
# Specify the libraries to link #
#################################

TARGET_LINK_LIBRARIES(${targetname}
	LNTLib
	ORUtils
)

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkPNG.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkCUDNN.cmake)
