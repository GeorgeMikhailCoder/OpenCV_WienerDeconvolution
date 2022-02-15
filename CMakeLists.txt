# CMakeList.txt : CMake project for OpenCV_WienerDeconvolution, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("OpenCV_WienerDeconvolution")

# Add source to this project's executable.
add_executable (OpenCV_WienerDeconvolution "OpenCV_WienerDeconvolution.cpp" "OpenCV_WienerDeconvolution.h")

# TODO: Add tests and install targets if needed.
find_package(OpenCV REQUIRED)
target_link_libraries(OpenCV_WienerDeconvolution ${OpenCV_LIBS})