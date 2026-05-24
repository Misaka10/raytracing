cmake_minimum_required(VERSION 3.14)
project(rt-next-week LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include external)

add_executable(rt src/main.cc)

# Standalone Monte Carlo demo programs
add_executable(pi src/pi.cc)
add_executable(integrate_x_sq src/integrate_x_sq.cc)
add_executable(cos_cubed src/cos_cubed.cc)
add_executable(cos_density src/cos_density.cc)
add_executable(sphere_importance src/sphere_importance.cc)
add_executable(sphere_plot src/sphere_plot.cc)
add_executable(estimate_halfway src/estimate_halfway.cc)
