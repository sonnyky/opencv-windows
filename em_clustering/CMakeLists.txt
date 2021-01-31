# Define project name
project(em_clustering)

# Set installation folder
set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )

# Set additional include folders
set (ADDITIONAL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${ADDITIONAL_INCLUDE})

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
set(SOURCES main.cpp ${ADDITIONAL_INCLUDE}/util.h)
add_executable(em_clustering ${SOURCES} )

# Link your application with OpenCV libraries
target_link_libraries(em_clustering ${OpenCV_LIBS})