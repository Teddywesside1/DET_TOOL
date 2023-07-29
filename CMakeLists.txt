cmake_minimum_required(VERSION 3.10)
project(server)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


set(SERVER_PLATFORM 0)
set(WITHOUT_IMAGE_READING false)
set(LISTEN_PORT 11188)
set(DATASET_DIR "/media/orin/root/binfeng/coco-val2017-5000/")
set(POST_URL "/orin_request")

configure_file(CMakeConfig.h.in CMakeConfig.h)

if (WITHOUT_IMAGE_READING)
    message(STATUS "the project will be built without OpenCV library")
else()
    message(STATUS "Building with OpenCV library")
    find_package(OpenCV 4.0)
    if (OpenCV_FOUND)
        message(STATUS "Found OpenCV library !")
        message(STATUS "OpenCV_INCLUDE_DIRS : " ${OpenCV_INCLUDE_DIRS})
        message(STATUS "OpenCV_LIBS : " ${OpenCV_LIBS})
    else()
        message(STATUS "CMake Could not find OpenCV library, Using Default OpenCV library path !")
        FILE(GLOB OpenCV_LIBS /opt/apollo/sysroot/lib/libopencv*.so)
        set(OpenCV_INCLUDE_DIRS /opt/apollo/sysroot/include/opencv4)
    endif()
endif()

if (SERVER_PLATFORM MATCHES 0)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
else()
    set(HDPL_TOOLCHAIN_ITVM_INSTALL "/home/binfeng.zhang/platform/toolchain/itvm/build/install")
    set(HDPL_INSTALL_PATH $ENV{HDPL_PATH})
    include_directories(${HDPL_INSTALL_PATH}/include)
    include_directories(${HDPL_TOOLCHAIN_ITVM_INSTALL}/include)
endif()


find_package(glog REQUIRED)

add_subdirectory(common)
add_subdirectory(interfaces)
# add_subdirectory(model_instance)
# add_subdirectory(gpu_status)
# add_subdirectory(crow)
# add_subdirectory(inference_runtime)
add_subdirectory(dataloader)
add_subdirectory(models)
add_subdirectory(model_framework)
add_subdirectory(process_strategy)
add_subdirectory(tools)

enable_testing()
add_subdirectory(gtest)



# add_executable(server src/main.cpp src/models_manager.cpp)
# target_link_libraries(server PUBLIC crow)
# target_link_libraries(server PUBLIC inference_runtime)
# if (WITHOUT_IMAGE_READING)
#     target_link_libraries(server PUBLIC dataloader_fake)
# else()
#     target_link_libraries(server PUBLIC dataloader_offline)
#     target_link_libraries(server PUBLIC dataloader_online)
# endif()
# target_link_libraries(server PUBLIC yolov3)
# target_link_libraries(server PUBLIC resnet50)
# target_link_libraries(server PUBLIC -lpthread)
# target_link_libraries(server PUBLIC interfaces)


# if (SERVER_PLATFORM MATCHES 0)
#     target_link_libraries(server PUBLIC jetson_status)
#     target_link_libraries(server PUBLIC tensorrt_model)
# else()
#     target_link_libraries(server PUBLIC temp)
#     target_link_libraries(server PUBLIC hm800_model)
#     target_link_libraries(server PUBLIC hm800_status)
# endif()


# target_include_directories(server PUBLIC 
#     "${PROJECT_BINARY_DIR}"
# )



