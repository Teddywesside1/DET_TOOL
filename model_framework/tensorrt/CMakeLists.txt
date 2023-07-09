project(tensorrt)

FILE(GLOB tensorrt_src *.cpp)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(tensorrt ${tensorrt_src})

target_link_libraries(tensorrt ${CUDA_LIBRARIES})
target_link_libraries(tensorrt nvinfer)
target_link_libraries(tensorrt nvonnxparser)
target_link_libraries(tensorrt glog)
target_link_libraries(tensorrt interfaces)
target_link_libraries(tensorrt common)

target_include_directories(tensorrt INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})