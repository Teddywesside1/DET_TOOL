project(gtests)

FILE(GLOB gtests_src *.cpp)

# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})

find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

add_executable(gtests ${gtests_src})

# target_link_libraries(tensorrt ${CUDA_LIBRARIES})
# target_link_libraries(tensorrt nvinfer)
# target_link_libraries(tensorrt nvonnxparser)
target_link_libraries(gtests glog)
target_link_libraries(gtests gtest)
target_link_libraries(gtests interfaces)
target_link_libraries(gtests tensorrt)
target_link_libraries(gtests yolov5)
target_link_libraries(gtests dataloader_objdet_2d)
target_link_libraries(gtests GTest::Main)
target_link_libraries(gtests common)

target_include_directories(gtests INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

