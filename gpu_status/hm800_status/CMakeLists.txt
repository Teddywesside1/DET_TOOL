project(hm800_status)

FILE(GLOB hm800_status_src *.cpp)

add_library(hm800_status ${hm800_status_src})

# target_link_libraries(hm800_status pthread)
target_link_libraries(hm800_status interfaces)
target_include_directories(hm800_status INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})