project(stereo3d)

FILE(GLOB stereo3d_src *.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(stereo3d ${yolov3_src})



target_link_libraries(stereo3d ${OpenCV_LIBS})
target_include_directories(stereo3d INTERFACE ${OpenCV_INCLUDE_DIRS})


target_link_libraries(yolov3 inference_runtime)
target_include_directories(yolov3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(inference_runtime PUBLIC ${CMAKE_BINARY_DIR})