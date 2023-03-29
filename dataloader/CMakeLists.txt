
project(dataloader)

if (WITHOUT_IMAGE_READING)
    add_subdirectory(fake)
else()
    add_subdirectory(offline)
    add_subdirectory(online)
endif()