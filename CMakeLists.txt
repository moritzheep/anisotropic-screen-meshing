cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_subdirectory(external)

add_subdirectory(src)
