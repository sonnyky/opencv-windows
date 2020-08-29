cmake_minimum_required(VERSION 2.8)
project (opencv_windows)
 
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
 
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_subdirectory(face_zoom)
add_subdirectory(face_recognition)
add_subdirectory(capture_faces)
add_subdirectory(draw_rect)
add_subdirectory(webcam_to_video_file)
add_subdirectory(up_base)
add_subdirectory(detect_brightness)
add_subdirectory(face_detection)
add_subdirectory(cylinder_detection)