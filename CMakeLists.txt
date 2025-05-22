# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# pybind and eigen require CXX >= 14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(C_STANDARD 17)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX C)

# This make will compile assist and rebound
execute_process(
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/assist/src
)

add_custom_target(
  symlink_libs
  ALL COMMAND . ${CMAKE_CURRENT_SOURCE_DIR}/create_lib_links.sh
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

execute_process(
  COMMAND cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/autodiff
)

include_directories(
  include/eigen
  include/assist/src
  include/rebound/src
  include/autodiff
)

link_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/assist/src
  ${CMAKE_CURRENT_SOURCE_DIR}include/rebound/src
)

# Discover the assist library after building
add_library(
  assist
  STATIC
  IMPORTED
)
set_target_properties(
  assist PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/include/assist/src/libassist.so
)

# Discover the rebound library after building
add_library(
  rebound
  STATIC
  IMPORTED
)
set_target_properties(
  rebound PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/include/assist/src/librebound.so
)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE src/main.cpp WITH_SOABI)
# link all the required libraries
target_link_libraries(_core PRIVATE pybind11::headers assist rebound)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION _layup_cpp)