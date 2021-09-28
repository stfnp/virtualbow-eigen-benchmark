cmake_minimum_required(VERSION 3.10.2)
project(benchmark)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# External libraries

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

#include_directories(source)

add_executable(
    benchmark
    source/main.cpp
)

target_link_libraries(
    benchmark
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)
