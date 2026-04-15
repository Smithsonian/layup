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

# Find platform-specific suffix for shared libraries
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "import importlib.machinery as m; print(m.EXTENSION_SUFFIXES[0])"
  OUTPUT_VARIABLE EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "import assist, os; print(os.path.dirname(assist.__file__))"
  OUTPUT_VARIABLE ASSIST_PKG_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ASSIST_RESULT
)
if(NOT ASSIST_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not find assist package. Is assist installed in the build environment?")
endif()

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c
          "import rebound, os; print(os.path.dirname(rebound.__file__))"
  OUTPUT_VARIABLE REBOUND_PKG_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE REBOUND_RESULT
)
if(NOT REBOUND_RESULT EQUAL 0)
  message(FATAL_ERROR "Could not find rebound package. Is rebound installed in the build environment?")
endif()

# libassist.so and librebound.so sit in site-packages, one level up from the package dir
cmake_path(GET ASSIST_PKG_DIR PARENT_PATH ASSIST_SITE_PACKAGES)
cmake_path(GET REBOUND_PKG_DIR PARENT_PATH REBOUND_SITE_PACKAGES)

file(GLOB LIBASSIST_SO "${ASSIST_SITE_PACKAGES}/libassist${EXT_SUFFIX}")
file(GLOB LIBREBOUND_SO "${REBOUND_SITE_PACKAGES}/librebound${EXT_SUFFIX}")

if(NOT LIBASSIST_SO)
  message(FATAL_ERROR "libassist${EXT_SUFFIX} not found in ${ASSIST_SITE_PACKAGES}")
endif()
if(NOT LIBREBOUND_SO)
  message(FATAL_ERROR "librebound${EXT_SUFFIX} not found in ${REBOUND_SITE_PACKAGES}")
endif()

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

if(APPLE)
  set(_LAYUP_RUNTIME_RPATH "@loader_path/..")
elseif(UNIX)
  set(_LAYUP_RUNTIME_RPATH "$ORIGIN/..")
endif()

if(DEFINED _LAYUP_RUNTIME_RPATH)
  set_target_properties(_core PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    BUILD_RPATH "${_LAYUP_RUNTIME_RPATH}"
    INSTALL_RPATH "${_LAYUP_RUNTIME_RPATH}"
  )
endif()

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION _layup_cpp)