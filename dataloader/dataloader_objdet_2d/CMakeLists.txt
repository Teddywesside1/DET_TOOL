
project(dataloader_objdet_2d)

FILE(GLOB dataloader_objdet_2d_src *.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
# FILE(GLOB OpenCV_LIBS /opt/apollo/sysroot/lib/libopencv*.so)


add_library(dataloader_objdet_2d ${dataloader_objdet_2d_src})

# target_link_libraries(dataloader_objdet_2d ${OpenCV_LIBS})
target_link_libraries(dataloader_objdet_2d interfaces)

target_include_directories(dataloader_objdet_2d INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
# target_include_directories(dataloader_offline INTERFACE ${OpenCV_INCLUDE_DIRS})