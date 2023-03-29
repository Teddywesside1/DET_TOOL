
project(dataloader_fake)

FILE(GLOB dataloader_fake_src *.cpp)

add_library(dataloader_fake ${dataloader_fake_src})

target_link_libraries(dataloader_fake inference_runtime)
target_link_libraries(dataloader_fake interfaces)
target_link_libraries(dataloader_fake common)

target_include_directories(dataloader_fake INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
