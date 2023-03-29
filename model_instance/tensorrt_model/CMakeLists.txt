
project(tensorrt_model)

FILE(GLOB tensorrt_model_src *.cpp)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(tensorrt_model ${tensorrt_model_src})

target_link_libraries(tensorrt_model ${CUDA_LIBRARIES})
target_link_libraries(tensorrt_model nvinfer)
target_link_libraries(tensorrt_model nvonnxparser)
target_link_libraries(tensorrt_model inference_runtime)

target_include_directories(tensorrt_model INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})