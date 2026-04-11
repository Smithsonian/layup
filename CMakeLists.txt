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

# Locate site-packages and determine arch-specific extension for the libraries
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
  "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE SITE_PACKAGES_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
  "import importlib.machinery as m; print(m.EXTENSION_SUFFIXES[0])"
  OUTPUT_VARIABLE EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Assist and rebound are installed as regular packages, so their library
# files will end up in site-packages.
set(LIBASSIST_SO "${SITE_PACKAGES_PATH}/libassist${EXT_SUFFIX}")
set(LIBREBOUND_SO "${SITE_PACKAGES_PATH}/librebound${EXT_SUFFIX}")

execute_process(
  COMMAND cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/autodiff
)

include_directories(
  include/eigen
  include/autodiff
)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE src/main.cpp WITH_SOABI)
# link all the required libraries
target_link_libraries(_core PRIVATE pybind11::headers ${LIBASSIST_SO} ${LIBREBOUND_SO})

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION _layup_cpp)