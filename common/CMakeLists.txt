project(common)

FILE(GLOB common_src *.cpp)

add_library(common ${common_src})

target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})