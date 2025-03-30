cmake_minimum_required(VERSION 3.5.0)
project(NumCpp VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

include_directories(./include)

aux_source_directory(./src/shaped shaped_src)
aux_source_directory(./src/utils utils_src)
aux_source_directory(./src/autograd autograd_src)
aux_source_directory(./src/utils/matmul utils_src_mat)
add_executable(NumCpp ./src/main.cpp ${shaped_src} ${utils_src} ${utils_src_mat} ${autograd_src})
# 引入fmt、pthread
add_subdirectory(libs/fmt EXCLUDE_FROM_ALL)
include_directories(libs/fmt/include)
target_link_libraries(NumCpp fmt Threads::Threads)

