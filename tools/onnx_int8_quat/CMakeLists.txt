project(tool_onnx_int8_quat)

FILE(GLOB tool_onnx_int8_quat_src *.cpp)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(tool_onnx_int8_quat ${tool_onnx_int8_quat_src})

target_link_libraries(tool_onnx_int8_quat ${CUDA_LIBRARIES})
target_link_libraries(tool_onnx_int8_quat nvinfer)
target_link_libraries(tool_onnx_int8_quat nvonnxparser)
target_link_libraries(tool_onnx_int8_quat glog)
target_link_libraries(tool_onnx_int8_quat common)
target_link_libraries(tool_onnx_int8_quat tensorrt)
target_link_libraries(tool_onnx_int8_quat process_strategy)

target_include_directories(tool_onnx_int8_quat INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})