project(gpu_status)

if (SERVER_PLATFORM MATCHES 0)
    add_subdirectory(jetson_status)
else()
    add_subdirectory(hm800_status)
endif()