project(interfaces)


include_directories(${OpenCV_INCLUDE_DIRS})

add_library(interfaces INTERFACE)

target_link_libraries(interfaces INTERFACE ${OpenCV_LIBS})
target_include_directories(interfaces INTERFACE ${OpenCV_INCLUDE_DIRS})

target_include_directories(interfaces INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(interfaces INTERFACE ${CMAKE_BINARY_DIR})