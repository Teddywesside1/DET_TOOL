project(process_strategy)

FILE(GLOB process_strategy_src *.cpp)

add_library(process_strategy ${process_strategy_src})
target_link_libraries(process_strategy interfaces)

target_include_directories(process_strategy INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})