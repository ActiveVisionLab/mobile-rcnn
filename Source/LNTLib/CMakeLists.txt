#############################
# CMakeLists.txt for LNTLib #
#############################

###########################
# Specify the target name #
###########################

SET(targetname LNTLib)

################################
# Specify the libraries to use #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake)
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseCUDNN.cmake)

#############################
# Specify the project files #
#############################

##
SET(LNTLIB_TOPLEVEL_HEADERS
LNTLib.h
)

##
SET(LNTLIB_CORE_HEADERS
Core/Connection.h
Core/Device.h
Core/Network.h
Core/Node.h
Core/TensorInfo.h
)

SET(LNTLIB_CORE_SOURCES
Core/Network.cpp
)

##
SET(LNTLIB_NETWORKS_HEADERS
Networks/MobileRCNN.h
)

SET(LNTLIB_NETWORKS_SOURCES
Networks/MobileRCNN.cpp
)

##
SET(LNTLIB_NODES_HEADERS
Nodes/BatchNorm.h
Nodes/Bias.h
Nodes/Composite.h
Nodes/Concat.h
Nodes/Conv.h
Nodes/ConvTransposed.h
Nodes/CopyOutput.h
Nodes/Crop.h
Nodes/ImageInput.h
Nodes/ImageOutput.h
Nodes/ImageResize.h
Nodes/InnerProduct.h
Nodes/Norm.h
Nodes/Pooling.h
Nodes/Power.h
Nodes/Relu.h
Nodes/Reshape.h
Nodes/RoIAlign.h
Nodes/Sigmoid.h
Nodes/Softmax.h
Nodes/Sum.h
Nodes/Upsample.h
)

SET(LNTLIB_NODES_IMPLEMENTATION_HEADERS
Nodes/Implementation/Implementation.h
)

SET(LNTLIB_NODES_IMPLEMENTATION_CUDNN_HEADERS
Nodes/Implementation/CUDNN/BatchNorm_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Bias_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Conv_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/ConvTransposed_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/CopyOutput_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Crop_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/ImageInput_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Implementation_CUDNN.h
Nodes/Implementation/CUDNN/InnerProduct_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Pooling_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Power_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Relu_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Reshape_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/RoIAlign_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Sigmoid_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Softmax_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Sum_Implementation_CUDNN.h
Nodes/Implementation/CUDNN/Upsample_Implementation_CUDNN.h
)

SET(LNTLIB_NODES_IMPLEMENTATION_CUDNN_SOURCES
Nodes/Implementation/CUDNN/BatchNorm_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Bias_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Conv_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/ConvTransposed_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/CopyOutput_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Crop_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/ImageInput_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/InnerProduct_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Pooling_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Power_Implementation_CUDNN.cu
Nodes/Implementation/CUDNN/Relu_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Reshape_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/RoIAlign_Implementation_CUDNN.cu
Nodes/Implementation/CUDNN/Sigmoid_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Softmax_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Sum_Implementation_CUDNN.cpp
Nodes/Implementation/CUDNN/Upsample_Implementation_CUDNN.cu
)

#################################################################
# Collect the project files into sources, headers and templates #
#################################################################

SET(sources
${LNTLIB_CORE_SOURCES}
${LNTLIB_NETWORKS_SOURCES}
${LNTLIB_NODES_IMPLEMENTATION_CUDNN_SOURCES}
)

SET(headers
${LNTLIB_TOPLEVEL_HEADERS}
${LNTLIB_CORE_HEADERS}
${LNTLIB_NETWORKS_HEADERS}
${LNTLIB_NODES_HEADERS}
${LNTLIB_NODES_IMPLEMENTATION_HEADERS}
${LNTLIB_NODES_IMPLEMENTATION_CUDNN_HEADERS}
)

#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${LNTLIB_TOPLEVEL_HEADERS})
SOURCE_GROUP(Core FILES ${LNTLIB_CORE_HEADERS} ${LNTLIB_CORE_SOURCES})
SOURCE_GROUP(Networks FILES ${LNTLIB_NETWORKS_SOURCES} ${LNTLIB_NETWORKS_HEADERS})
SOURCE_GROUP(Nodes FILES ${LNTLIB_NODES_HEADERS})
SOURCE_GROUP(Nodes\\Implementation FILES ${LNTLIB_NODES_IMPLEMENTATION_HEADERS})
SOURCE_GROUP(Nodes\\Implementation\\CPU FILES ${LNTLIB_NODES_IMPLEMENTATION_CPU_SOURCES} ${LNTLIB_NODES_IMPLEMENTATION_CPU_HEADERS})
SOURCE_GROUP(Nodes\\Implementation\\CUDNN FILES ${LNTLIB_NODES_IMPLEMENTATION_CUDNN_SOURCES} ${LNTLIB_NODES_IMPLEMENTATION_CUDNN_HEADERS})

##########################################
# Specify the target and where to put it #
##########################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/SetGPULibTarget.cmake)

IF(WITH_CUDNN)
	INCLUDE(${PROJECT_SOURCE_DIR}/cmake/LinkCUDNN.cmake)
ENDIF()
