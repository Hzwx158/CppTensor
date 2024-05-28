cmake_minimum_required(VERSION 3.5.0)
project(numHCpp VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

aux_source_directory(./shapedarray tensor_src)
# aux_source_directory(./autograd autograd_src)
include_directories("./shapedarray")
# include_directories("./autograd")

add_executable(numHCpp main.cpp ${tensor_src}) #${autograd_src})
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
