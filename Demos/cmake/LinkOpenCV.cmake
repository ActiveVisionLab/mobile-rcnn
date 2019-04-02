####################
# LinkOpenCV.cmake #
####################

IF(WITH_OPENCV)
  TARGET_LINK_LIBRARIES(${targetname} ${OpenCV_LIBS})

	IF(MSVC_IDE)
		FILE(GLOB RUNTIMELIBS "${OpenCV_DIR}/x64/vc15/bin/*.dll")

		FOREACH(RUNTIMELIB ${RUNTIMELIBS})
			ADD_CUSTOM_COMMAND(TARGET ${targetname} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RUNTIMELIB}" "$<TARGET_FILE_DIR:${targetname}>")
		ENDFOREACH()
	ENDIF()
ENDIF()
