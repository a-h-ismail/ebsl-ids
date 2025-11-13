cmake_minimum_required(VERSION 3.15...3.27)
project(ebsl_ids) # Replace 'my_project' with the name of your project

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(ebsl_cpp src/ebsl.cpp src/binomial_opinion.cpp src/bsl_sm.cpp)

nanobind_add_stub(
ebsl_cpp_stub
MODULE ebsl_cpp
OUTPUT ebsl_cpp.pyi
PYTHON_PATH $<TARGET_FILE_DIR:ebsl_cpp>
DEPENDS ebsl_cpp)

