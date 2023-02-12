cmake_minimum_required(VERSION 3.24)
project(Wordle5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
add_executable(wordle5_omp wordle5.cpp)
target_link_libraries(wordle5_omp PUBLIC OpenMP::OpenMP_CXX)
configure_file(words_alpha.txt words_alpha.txt COPYONLY)