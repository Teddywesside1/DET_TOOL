cmake_minimum_required(VERSION 3.10)
project(inference_runtime)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

FILE(GLOB inference_runtime_src *.cpp)

if (WITHOUT_IMAGE_READING)

else()
    include_directories(${OpenCV_INCLUDE_DIRS})
    # FILE(GLOB OpenCV_LIBS /opt/apollo/sysroot/lib/libopencv*.so)
endif()

# add_subdirectory(model_instance)

add_library(inference_runtime ${inference_runtime_src})
if (WITHOUT_IMAGE_READING)

else()
    target_link_libraries(inference_runtime ${OpenCV_LIBS})
    target_include_directories(inference_runtime INTERFACE ${OpenCV_INCLUDE_DIRS})
endif()

target_link_libraries(inference_runtime -lpthread)
target_link_libraries(inference_runtime interfaces)
target_link_libraries(inference_runtime common)

target_include_directories(inference_runtime INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(inference_runtime PUBLIC ${CMAKE_BINARY_DIR})