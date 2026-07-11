cmake_minimum_required(VERSION 3.16)

project(tesis_pose VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(tesis_pose
    src/main.cpp
    src/camera.cpp
    src/fps_meter.cpp
    src/csv_logger.cpp
)

target_include_directories(tesis_pose PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tesis_pose PRIVATE
    ${OpenCV_LIBS}
)

target_compile_options(tesis_pose PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
)

