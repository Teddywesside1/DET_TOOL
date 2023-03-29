cmake_minimum_required(VERSION 3.10)
project(CROW)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Boost COMPONENTS thread system REQUIRED)

link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

add_library(crow crow_all.cpp)

target_link_libraries(crow PUBLIC ${Boost_LIBRARIES})

target_include_directories(crow INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})