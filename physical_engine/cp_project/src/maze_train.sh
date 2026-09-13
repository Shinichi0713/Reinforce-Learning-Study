!pip install -q onnx onnxscript
!g++ -O3 maze_rl.cpp -o maze_rl
!./maze_rl


%%bash
# コンパイル
g++ -O3 cartpole_control.cpp -o cartpole_control \
    -I./onnxruntime/include \
    -L./onnxruntime/lib \
    -lonnxruntime -std=c++17

# 実行
LD_LIBRARY_PATH=./onnxruntime/lib ./cartpole_control