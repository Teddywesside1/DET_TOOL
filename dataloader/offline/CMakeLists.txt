
project(dataloader_offline)

FILE(GLOB dataloader_offline_src *.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
# FILE(GLOB OpenCV_LIBS /opt/apollo/sysroot/lib/libopencv*.so)


add_library(dataloader_offline ${dataloader_offline_src})

target_link_libraries(dataloader_offline -lpthread)
target_link_libraries(dataloader_offline ${OpenCV_LIBS})
target_link_libraries(dataloader_offline inference_runtime)

target_include_directories(dataloader_offline INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(dataloader_offline INTERFACE ${OpenCV_INCLUDE_DIRS})