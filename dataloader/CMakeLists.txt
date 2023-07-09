
project(dataloader)

if (WITHOUT_IMAGE_READING)
    add_subdirectory(fake)
else()
    add_subdirectory(offline)
    add_subdirectory(online)
    add_subdirectory(dataloader_objdet_2d)
endif()