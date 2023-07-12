project(yolov5)

FILE(GLOB yolov5_src *.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})


add_library(yolov5 ${yolov5_src})



target_link_libraries(yolov5 ${OpenCV_LIBS})
target_include_directories(yolov5 INTERFACE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolov5 glog)
target_link_libraries(yolov5 interfaces)
target_link_libraries(yolov5 common)
target_link_libraries(yolov5 process_strategy)

target_include_directories(yolov5 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(yolov5 PUBLIC ${CMAKE_BINARY_DIR})