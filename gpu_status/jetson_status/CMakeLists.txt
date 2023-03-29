project(jetson_status)

FILE(GLOB jetson_status_src *.cpp)

add_library(jetson_status ${jetson_status_src})

target_link_libraries(jetson_status pthread)
target_link_libraries(jetson_status interfaces)
target_include_directories(jetson_status INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})