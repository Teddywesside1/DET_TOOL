project(temp)

FILE(GLOB temp_src *.cpp)

add_library(temp ${temp_src})

target_link_libraries(temp inference_runtime)
target_link_libraries(temp interfaces)
target_include_directories(temp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
