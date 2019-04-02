####################
# LinkOpenCL.cmake #
####################

IF(WITH_CUDNN AND WITH_CUDA)
	TARGET_LINK_LIBRARIES(${targetname} ${CUDNN_LIBRARY})
	
	IF(MSVC_IDE)
		ADD_CUSTOM_COMMAND(TARGET ${targetname} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUDNN_ROOT_DIR}/bin/cudnn64_7.dll" "$<TARGET_FILE_DIR:${targetname}>")
	ENDIF()
ENDIF()