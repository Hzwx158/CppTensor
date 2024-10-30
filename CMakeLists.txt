cmake_minimum_required(VERSION 3.5.0)
project(NumCpp VERSION 0.1.0 LANGUAGES C CXX)


aux_source_directory(./shaped shaped_src)
include_directories("./shaped")
aux_source_directory(./utils utils_src)
include_directories("./utils")
add_executable(NumCpp main.cpp ${shaped_src} ${utils_src})

