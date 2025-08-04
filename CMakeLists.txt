cmake_minimum_required(VERSION 3.1)
project(pedestrians)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(include)


file(GLOB SRC_FILES src/*.cpp)

add_executable(pedestrians main.cpp ${SRC_FILES})

target_link_libraries(pedestrians ${OpenCV_LIBS})
target_include_directories(pedestrians PRIVATE ${OpenCV_INCLUDE_DIRS})
