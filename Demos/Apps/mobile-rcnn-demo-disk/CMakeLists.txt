#################################################
# CMakeLists.txt for Apps/mobile-rcnn-demo-disk #
#################################################

###########################
# Specify the target name #
###########################

SET(targetname mobile-rcnn-demo-disk)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseLNTLibRedist.cmake)

#############################
# Specify the project files #
#############################

SET(sources
MobileRCNN.cpp
FileUtils.cpp
)

SET(headers
FileUtils.h
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${sources} ${headers})

##########################################
# Specify the target and where to put it #
##########################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/SetAppTarget.cmake)

#################################
# Specify the libraries to link #
#################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkLNTLibRedist.cmake)