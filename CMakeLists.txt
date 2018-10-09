project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall -g")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources src/particle_filter.cpp src/main.cpp)


add_executable(particle_filter ${sources})


