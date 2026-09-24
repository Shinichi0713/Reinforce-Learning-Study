cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(dqn_cpp)

set(CMAKE_CXX_STANDARD 17)

# LibTorchのパスを指定（環境に合わせて変更してください）
# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)

add_executable(dqn_cpp main.cpp)
target_link_libraries(dqn_cpp "${TORCH_LIBRARIES}")
set_property(TARGET dqn_cpp PROPERTY CXX_STANDARD 17)