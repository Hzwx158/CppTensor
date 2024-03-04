cmake_minimum_required(VERSION 3.5.0)
project(numHCpp VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

aux_source_directory(./tensor tensor_src)
include_directories("./tensor")

add_executable(numHCpp main.cpp ${tensor_src})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
