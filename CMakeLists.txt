cmake_minimum_required(VERSION 3.24)
project(Wordle5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (UNIX)
message("Compiling on Unix")
add_compile_options(-march=native -Ofast)
endif(UNIX)

find_package(OpenMP)
add_executable(wordle5_omp wordle5.cpp)
target_link_libraries(wordle5_omp PUBLIC OpenMP::OpenMP_CXX)

add_executable(oisyn oisyn.cpp)
target_link_libraries(oisyn PUBLIC OpenMP::OpenMP_CXX)

configure_file(words_alpha.txt words_alpha.txt COPYONLY)