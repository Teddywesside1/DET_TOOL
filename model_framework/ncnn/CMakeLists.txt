project(framework_ncnn)

FILE(GLOB framework_ncnn_src *.cpp)

FIND_PACKAGE(OpenMP REQUIRED)  
if(OPENMP_FOUND)  
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

find_package(Vulkan REQUIRED)
# find_package(glslang REQUIRED)

include_directories("/data/binfeng/projects/ncnn/build/install/include/ncnn")
FILE(GLOB ncnn_LIBS "/data/binfeng/projects/ncnn/build/install/lib/lib*")
MESSAGE(STATUS "ncnn_libs : " ${ncnn_LIBS})


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- ncnn
# set函数里的路径是指包含ncnnConfig.cmake的目录
# set(ncnn_DIR "/data/binfeng/projects/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "/data/binfeng/projects/ncnn/build/install/lib/cmake/ncnn")
# find_package(ncnn REQUIRED)

add_library(framework_ncnn ${framework_ncnn_src})

target_link_libraries(framework_ncnn ${ncnn_LIBS})
target_link_libraries(framework_ncnn glog)
target_link_libraries(framework_ncnn interfaces)
target_link_libraries(framework_ncnn common)
target_link_libraries(framework_ncnn OpenMP::OpenMP_CXX)
target_link_libraries(framework_ncnn Vulkan::Vulkan)
# target_link_libraries(framework_ncnn glslang::glslang)

target_include_directories(framework_ncnn INTERFACE "/data/binfeng/projects/ncnn/build/install/include/ncnn")
target_include_directories(framework_ncnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})