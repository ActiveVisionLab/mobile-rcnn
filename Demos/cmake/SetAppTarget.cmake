######################
# SetAppTarget.cmake #
######################

INCLUDE(${PROJECT_SOURCE_DIR}/cmake/Flags.cmake)

ADD_EXECUTABLE(${targetname} ${sources} ${headers} ${templates})

IF(MSVC_IDE)
  SET_TARGET_PROPERTIES(${targetname} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG")
ENDIF()
