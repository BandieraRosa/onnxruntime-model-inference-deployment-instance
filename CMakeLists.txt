cmake_minimum_required(VERSION 2.5)
 
project(yolo_detector CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
 
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT_USER_PATH ./onnxruntime-linux-x64-1.22.0)
get_filename_component(ONNXRUNTIME_ROOT_PATH_ABS ${ONNXRUNTIME_ROOT_USER_PATH} ABSOLUTE)

set(ONNXRUNTIME_SINGLE_INCLUDE_DIR ${ONNXRUNTIME_ROOT_PATH_ABS}/include)
set(ONNXRUNTIME_LIBRARY_FILE ${ONNXRUNTIME_ROOT_PATH_ABS}/lib/libonnxruntime.so)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_SINGLE_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY_FILE}
)