cmake_minimum_required(VERSION 3.16)
project(numeric_methods_lab_2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(task_2_1 task_2_1.cpp)
add_executable(task_2_2 task_2_2.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(task_2_1 PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(task_2_2 PRIVATE -Wall -Wextra -Wpedantic)
endif()
