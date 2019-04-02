##############################
# CMakeLists.txt for ORUtils #
##############################

###########################
# Specify the target name #
###########################
SET(targetname ORUtils)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDNN.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UsePNG.cmake)

#############################
# Specify the project files #
#############################

SET(sources
FileUtils.cpp
)

SET(headers
CUDADefines.h
Image.h
ImageTypes.h
MathTypes.h
MathUtils.h
Matrix.h
MemoryBlock.h
MemoryBlockTypes.h
MemoryDeviceDefs.h
NVTimer.h
PixelPrimitives.h
PlatformIndependence.h
TimeStampedImage.h
Vector.h
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP(Source FILES ${sources})
SOURCE_GROUP(Headers FILES ${headers})

##########################################
# Specify the target and where to put it #
##########################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/SetGPULibTarget.cmake)

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkPNG.cmake)