cmake_minimum_required(VERSION 3.16)
project(dmm LANGUAGES CXX)

add_executable(dmm src/main.cpp)
target_compile_features(dmm PRIVATE cxx_std_17)
set_target_properties(dmm PROPERTIES CXX_EXTENSIONS OFF)
