##########################
# SetCUDAAppTarget.cmake #
##########################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

IF(WITH_CUDA)
  CUDA_ADD_EXECUTABLE(${targetname} ${sources} ${headers} ${templates})
ELSE()
  ADD_EXECUTABLE(${targetname} ${sources} ${headers} ${templates})

  IF(WITH_METAL)
    SET_TARGET_PROPERTIES(${targetname} PROPERTIES XCODE_ATTRIBUTE_MTL_PREPROCESSOR_DEFINITIONS "__METALC__")
  ENDIF()

ENDIF()

IF(MSVC_IDE)
  SET_TARGET_PROPERTIES(${targetname} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG")
ENDIF()
