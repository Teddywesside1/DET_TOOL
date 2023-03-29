project(model_instance)

if (SERVER_PLATFORM MATCHES 0)
    add_subdirectory(tensorrt_model)
else()
    add_subdirectory(temp)
    add_subdirectory(hm800_model)
endif()