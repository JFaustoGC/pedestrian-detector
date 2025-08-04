cmake_minimum_required(VERSION 3.1)
project(pedestrians)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(include)


add_executable(pedestrians main.cpp src/descriptors.cpp src/utils.cpp src/io.cpp)
target_link_libraries(pedestrians ${OpenCV_LIBS})
target_include_directories(pedestrians PRIVATE ${OpenCV_INCLUDE_DIRS})
