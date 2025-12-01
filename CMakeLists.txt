cmake_minimum_required(VERSION 3.15...3.27)
project(ebsl_ids)

# Warn if the user invokes CMake directly
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

set(CMAKE_CXX_STANDARD 20)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)

nanobind_add_module(ebsl_cpp src/ebsl.cpp src/binomial_opinion.cpp src/bsl_sm.cpp)

nanobind_add_stub(
ebsl_cpp_stub
MODULE ebsl_cpp
OUTPUT ebsl_cpp.pyi
PYTHON_PATH $<TARGET_FILE_DIR:ebsl_cpp>
DEPENDS ebsl_cpp)

install(TARGETS ebsl_cpp LIBRARY DESTINATION ebsl)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ebsl_cpp.pyi DESTINATION ebsl)
