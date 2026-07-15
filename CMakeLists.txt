cmake_minimum_required(VERSION 3.16)

project(tesis_pose VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ONNXRUNTIME_DIR "/media/user/Respaldos/Descargas/onnxruntime")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

add_executable(tesis_pose
    src/main.cpp
    src/camera.cpp
    src/fps_meter.cpp
    src/csv_logger.cpp
    src/pose_estimator.cpp
)

target_include_directories(tesis_pose PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

target_link_directories(tesis_pose PRIVATE
    ${ONNXRUNTIME_DIR}/lib
)

target_link_libraries(tesis_pose PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

set_target_properties(tesis_pose PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_DIR}/lib"
)

target_compile_options(tesis_pose PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
)
