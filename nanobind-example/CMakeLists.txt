cmake_minimum_required(VERSION 3.15...3.27)
project(my_project) # Replace 'my_project' with the name of your project

if(CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

# set(Python3_EXECUTABLE /home/jeromeku/llvm-project/.venv/bin/python)
set(Python3_FIND_VIRTUALENV FIRST) # or ONLY, if you insist

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
message(STATUS "Python SOABI ${Python_SOABI}")
message(STATUS "Python INCLUDE ${Python_INCLUDE_DIRS}")

cmake_path(GET CMAKE_SOURCE_DIR PARENT_PATH PARENT_DIR)
message(STATUS "Parent directory: ${PARENT_DIR}")

add_subdirectory(/home/jeromeku/llvm-project/nanobind-example/ext/nanobind)

nanobind_add_module(my_ext NOMINSIZE my_ext.cpp)
target_compile_options(my_ext PRIVATE
    -O0
    -g
    -fno-omit-frame-pointer
)
