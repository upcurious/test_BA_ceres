cmake_minimum_required(VERSION 2.8.3)
set(PROJECT_NAME test_BA)
project(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)



add_executable(BA main.cpp)
target_link_libraries(BA ${OpenCV_LIBS} ${CERES_LIBRARIES} ${EIGEN_LIBRARIES})


