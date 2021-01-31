cmake_minimum_required(VERSION 3.10)
project (opencv_windows)
 
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set( OpenCV_DIR "C:/Users/Sonny/Desktop/Libraries/opencvBuild")
option( OpenCV_STATIC OFF )
find_package(OpenCV REQUIRED)

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

#add_subdirectory(face_zoom)
#add_subdirectory(face_recognition)
add_subdirectory(capture_faces)
#add_subdirectory(draw_rect)
add_subdirectory(em_clustering)
#add_subdirectory(webcam_to_video_file)
#add_subdirectory(up_base)
#add_subdirectory(detect_brightness)
#add_subdirectory(face_detection)
#add_subdirectory(cylinder_detection)