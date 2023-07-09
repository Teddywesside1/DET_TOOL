project(common)

FILE(GLOB common_src *.cpp)

add_library(common ${common_src})
target_link_libraries(common glog)
target_link_libraries(common interfaces)

target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})