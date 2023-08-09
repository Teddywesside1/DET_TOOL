project(process_strategy)

FILE(GLOB process_strategy_src *.cpp)

find_package(OpenMP REQUIRED)

add_library(process_strategy ${process_strategy_src})
target_link_libraries(process_strategy interfaces)
target_link_libraries(process_strategy OpenMP::OpenMP_CXX)

target_include_directories(process_strategy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})