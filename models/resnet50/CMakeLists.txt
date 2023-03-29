project(resnet50)

FILE(GLOB resnet50_src *.cpp)


if (WITHOUT_IMAGE_READING)

else()
    include_directories(${OpenCV_INCLUDE_DIRS})
    # FILE(GLOB OpenCV_LIBS /opt/apollo/sysroot/lib/libopencv*.so)
endif()


add_library(resnet50 ${resnet50_src})

if (WITHOUT_IMAGE_READING)

else()
    target_link_libraries(resnet50 ${OpenCV_LIBS})
    target_include_directories(resnet50 INTERFACE ${OpenCV_INCLUDE_DIRS})
endif()

target_link_libraries(resnet50 inference_runtime)
target_include_directories(resnet50 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(inference_runtime PUBLIC ${CMAKE_BINARY_DIR})
