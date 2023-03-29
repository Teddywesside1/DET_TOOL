
project(dataloader_online)

FILE(GLOB dataloader_online_src *.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(/opt/apollo/sysroot/include/opencv4)
# FILE(GLOB OpenCV_LIBS /opt/apollo/sysroot/lib/libopencv*.so)

add_library(dataloader_online ${dataloader_online_src})

target_link_libraries(dataloader_online -lpthread)
target_link_libraries(dataloader_online ${OpenCV_LIBS})
target_link_libraries(dataloader_online inference_runtime)

target_include_directories(dataloader_online INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dataloader_online INTERFACE ${OpenCV_INCLUDE_DIRS})