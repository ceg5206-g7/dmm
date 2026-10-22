cmake_minimum_required(VERSION 3.21)
project(dmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)

add_library(dmm_core src/bsp.cpp src/cannon.cpp src/execution.cpp)
target_include_directories(dmm_core PUBLIC include)
target_link_libraries(dmm_core PUBLIC Threads::Threads)
target_compile_options(dmm_core PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(dmm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_native src/bindings.cpp)
target_link_libraries(_native PRIVATE dmm_core)
target_compile_definitions(_native PRIVATE DMM_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
install(TARGETS _native DESTINATION dmm)

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()
  # GoogleTest 1.17.0, pinned to its release commit and archive checksum.
  FetchContent_Declare(googletest
    URL https://codeload.github.com/google/googletest/zip/52eb8108c5bdec04579160ae17225d66034bd723
    DOWNLOAD_NAME googletest-v1.17.0.zip
    URL_HASH SHA256=9a56a54ae784394ff664cd55e8f4c9a03b503ebf0cb99576321c78ab3d87ca84)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(dmm_tests
    tests/cpp/test_gemm.cpp
    tests/cpp/test_matrix.cpp
    tests/cpp/test_matrix_block_view.cpp
    tests/cpp/test_bsp.cpp
    tests/cpp/test_cannon.cpp
    tests/cpp/test_executor.cpp)
  target_link_libraries(dmm_tests PRIVATE dmm_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(dmm_tests PROPERTIES TIMEOUT 30)
endif()
