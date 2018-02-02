# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(base_opencv)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set( OpenCV_DIR "D:/Libraries/opencv3.4.0/build" )
option( OpenCV_STATIC OFF )
find_package(OpenCV REQUIRED)

# Set installation folder
set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )

# Set additional include folders
set (ADDITIONAL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/src/include")

include_directories(${ADDITIONAL_INCLUDE})

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


# Declare the executable target built from your sources
set(SOURCES src/main.cpp src/app.cpp ${ADDITIONAL_INCLUDE}/app.h)
add_executable(base_opencv WIN32 ${SOURCES} )

if(WIN32)
  target_link_libraries(base_opencv wsock32 ws2_32 winmm)
endif()

# Link your application with OpenCV libraries
target_link_libraries(base_opencv ${OpenCV_LIBS})