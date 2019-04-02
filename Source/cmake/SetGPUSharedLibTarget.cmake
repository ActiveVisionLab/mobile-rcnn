################################
# SetCUDASharedLibTarget.cmake #
################################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

set(LIB_TYPE DLL)

IF(WITH_CUDA)
  CUDA_ADD_LIBRARY(${targetname} SHARED ${sources} ${headers} ${templates})
ELSE()
  ADD_LIBRARY(${targetname} SHARED ${sources} ${headers} ${templates})
  
  IF(WITH_METAL)
    SET_TARGET_PROPERTIES(${targetname} PROPERTIES XCODE_ATTRIBUTE_MTL_PREPROCESSOR_DEFINITIONS "__METALC__")
  ENDIF()

ENDIF()
