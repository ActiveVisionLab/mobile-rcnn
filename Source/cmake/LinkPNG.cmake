#################
# LinkPNG.cmake #
#################
IF(WITH_PNG)  
  IF(MSVC_IDE)
    ADD_CUSTOM_COMMAND(TARGET ${targetname} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PNG_ROOT}/bin/libpng16.dll" "$<TARGET_FILE_DIR:${targetname}>")
    ADD_CUSTOM_COMMAND(TARGET ${targetname} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PNG_ROOT}/bin/libpng16d.dll" "$<TARGET_FILE_DIR:${targetname}>")

    SET(PNG_LIBRARY optimized ${PNG_LIBRARY_RELEASE} debug ${PNG_LIBRARY_DEBUG} )
  ELSE()
  ENDIF()
  TARGET_LINK_LIBRARIES(${targetname} ${PNG_LIBRARY})
ENDIF()
