# Define project name
project(capture_faces)

# Set installation folder
set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )

# Set additional include folders
set (ADDITIONAL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/src/include")

include_directories(${PUGI_DIR})
include_directories(${ADDITIONAL_INCLUDE})

# Declare the executable target built from your sources
set(SOURCES src/main.cpp src/app.cpp ${PUGI_DIR}/pugixml.cpp ${ADDITIONAL_INCLUDE}/app.h)
add_executable(capture_faces ${SOURCES} )

if(WIN32)
  target_link_libraries(capture_faces wsock32 ws2_32 winmm)
endif()

# Link your application with OpenCV libraries
target_link_libraries(capture_faces ${OpenCV_LIBS})