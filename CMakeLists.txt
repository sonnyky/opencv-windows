cmake_minimum_required(VERSION 2.8)
project (opencv_windows)
 
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(face_zoom)
add_subdirectory(face_recognition)
add_subdirectory(capture_faces)