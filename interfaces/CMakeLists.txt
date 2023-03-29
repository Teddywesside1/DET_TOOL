project(interfaces)


if (WITHOUT_IMAGE_READING)

else()
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

add_library(interfaces INTERFACE)

if (WITHOUT_IMAGE_READING)

else()
    target_link_libraries(interfaces ${OpenCV_LIBS})
    target_include_directories(interfaces INTERFACE ${OpenCV_INCLUDE_DIRS})
endif()

target_include_directories(interfaces INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(interfaces INTERFACE ${CMAKE_BINARY_DIR})