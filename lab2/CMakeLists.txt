cmake_minimum_required(VERSION 3.11)
project(DGEMV C)

set(CMAKE_C_STANDARD 11)
link_libraries("-fopenmp -lxmalloc")

add_executable(DGEMV main.c)
