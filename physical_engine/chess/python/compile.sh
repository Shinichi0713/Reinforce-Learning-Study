!apt-get update -qq && apt-get install -y -qq libopencv-dev pkg-config

%%bash
# 1. 古い SDK の削除
rm -rf ./ort

# 2. 依存ライブラリの確認とインストール
apt-get update -qq && apt-get install -y -qq libopencv-dev pkg-config wget tar

# 3. ONNX Runtime v1.20.1 (IR Version 10 対応) のダウンロード
echo "Downloading ONNX Runtime C++ SDK v1.20.1..."
wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-1.20.1.tgz
mv onnxruntime-linux-x64-1.20.1 ort
rm onnxruntime-linux-x64-1.20.1.tgz

# 4. コンパイル
g++ -O3 main.cpp \
    -I./ort/include `pkg-config --cflags opencv4` \
    -L./ort/lib `pkg-config --libs opencv4` -lonnxruntime \
    -o cnn_inference

# 5. 実行
export LD_LIBRARY_PATH=./ort/lib:$LD_LIBRARY_PATH
./cnn_inference
