%%writefile main.cpp
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// CIFAR-10 の 10 クラスラベル
const std::vector<std::string> LABELS = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

int main() {
    std::string image_path = "test_image.png";
    std::string model_path = "cnn_cifar10.onnx";

    // ---------------------------------------------------------
    // 1. OpenCV による画像読み込みと前処理 (PyTorchと同等)
    // ---------------------------------------------------------
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    // BGR -> RGB 変換
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 32x32 にリサイズ
    cv::resize(img, img, cv::Size(32, 32));

    // float32 型へ変換し [0, 1] に正規化
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    // HWC (32, 32, 3) から NCHW (1, 3, 32, 32) への変換 & 標準化 (mean=0.5, std=0.5)
    // (x - 0.5) / 0.5
    std::vector<float> input_tensor_values(1 * 3 * 32 * 32);
    int spatial_size = 32 * 32;

    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 32; ++w) {
            cv::Vec3f pixel = img.at<cv::Vec3f>(h, w);
            for (int c = 0; c < 3; ++c) {
                // PyTorch transform: (val - 0.5) / 0.5
                float normalized_val = (pixel[c] - 0.5f) / 0.5f;
                // NCHW 形式のフラットインデックス計算
                int index = c * spatial_size + h * 32 + w;
                input_tensor_values[index] = normalized_val;
            }
        }
    }

    // ---------------------------------------------------------
    // 2. ONNX Runtime セッションの構築
    // ---------------------------------------------------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CNN_Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    Ort::Session session(env, model_path.c_str(), session_options);

    // ---------------------------------------------------------
    // 3. 入力 Tensor のバインド
    // ---------------------------------------------------------
    std::vector<int64_t> input_shape = {1, 3, 32, 32};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // ---------------------------------------------------------
    // 4. 推論の実行
    // ---------------------------------------------------------
    std::cout << "Running C++ Inference with ONNX Runtime..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // ---------------------------------------------------------
    // 5. 結果の解析 (ArgMax)
    // ---------------------------------------------------------
    float* logits = output_tensors[0].GetTensorMutableData<float>();
    size_t num_classes = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    // 最も高いスコアのクラスインデックスを取得
    int predicted_class = std::distance(logits, std::max_element(logits, logits + num_classes));

    std::cout << "\n=== 推論結果 ===" << std::endl;
    std::cout << "Predicted Class ID : " << predicted_class << std::endl;
    std::cout << "Predicted Label    : " << LABELS[predicted_class] << std::endl;
    std::cout << "Logit Score        : " << logits[predicted_class] << std::endl;

    return 0;
}