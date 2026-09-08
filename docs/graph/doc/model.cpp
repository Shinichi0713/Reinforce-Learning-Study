#include <torch/torch.h>
#include <iostream>
#include <vector>

// ============================================
// 1. 基本MLP（多層パーセプトロン）
// ============================================
struct MLP : torch::nn::Module {
    // 層の宣言
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::Dropout dropout{nullptr};

    MLP(int64_t input_size, int64_t hidden_size, int64_t num_classes)
        : fc1(register_module("fc1", torch::nn::Linear(input_size, hidden_size))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_size, hidden_size))),
          fc3(register_module("fc3", torch::nn::Linear(hidden_size, num_classes))),
          dropout(register_module("dropout", torch::nn::Dropout(0.5))) {}

    // 順伝播
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));   // 第1層 + ReLU
        x = torch::relu(fc2->forward(x));   // 第2層 + ReLU
        x = dropout->forward(x);             // Dropout
        x = fc3->forward(x);                 // 出力層（生のlogit）
        return x;
    }
};

// ============================================
// 2. CNN（畳み込みニューラルネット）— MNIST想定
// ============================================
struct CNN : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::MaxPool2d pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    CNN()
        : conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)))),
          bn1(register_module("bn1", torch::nn::BatchNorm2d(32))),
          conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)))),
          bn2(register_module("bn2", torch::nn::BatchNorm2d(64))),
          pool(register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)))),
          fc1(register_module("fc1", torch::nn::Linear(64 * 7 * 7, 128))),
          fc2(register_module("fc2", torch::nn::Linear(128, 10))) {}

    torch::Tensor forward(torch::Tensor x) {
        // 畳み込みブロック1: [N, 1, 28, 28] -> [N, 32, 28, 28] -> [N, 32, 14, 14]
        x = pool->forward(torch::relu(bn1->forward(conv1->forward(x))));
        // 畳み込みブロック2: [N, 32, 14, 14] -> [N, 64, 14, 14] -> [N, 64, 7, 7]
        x = pool->forward(torch::relu(bn2->forward(conv2->forward(x))));

        // フラット化: [N, 64, 7, 7] -> [N, 64*7*7]
        x = x.view({-1, 64 * 7 * 7});

        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
};

// ============================================
// 3. ResNet風 残差ブロック（短絡接続）
// ============================================
struct ResidualBlock : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    int64_t in_channels, out_channels;
    bool upsample_needed;

    torch::nn::Sequential downsample{nullptr};

    ResidualBlock(int64_t in_ch, int64_t out_ch, int64_t stride = 1)
        : in_channels(in_ch), out_channels(out_ch),
          conv1(register_module("conv1", torch::nn::Conv2d(
              torch::nn::Conv2dOptions(in_ch, out_ch, 3).stride(stride).padding(1).bias(false)))),
          bn1(register_module("bn1", torch::nn::BatchNorm2d(out_ch))),
          conv2(register_module("conv2", torch::nn::Conv2d(
              torch::nn::Conv2dOptions(out_ch, out_ch, 3).stride(1).padding(1).bias(false)))),
          bn2(register_module("bn2", torch::nn::BatchNorm2d(out_ch))) {

        // チャネル数や解像度が変わる場合、短絡接続側も調整
        if (stride != 1 || in_ch != out_ch) {
            downsample = register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 1).stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_ch)
            ));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;

        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);

        out = conv2->forward(out);
        out = bn2->forward(out);

        // 短絡接続（Skip connection）
        if (downsample) {
            identity = downsample->forward(x);
        }

        out += identity;           // F(x) + x
        out = torch::relu(out);
        return out;
    }
};

// ============================================
// 4. Sequential を使った簡潔な書き方
// ============================================
torch::nn::Sequential make_simple_cnn() {
    using namespace torch::nn;
    return Sequential(
        Conv2d(Conv2dOptions(1, 16, 3).padding(1)),
        BatchNorm2d(16),
        Functional(torch::relu),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Conv2d(Conv2dOptions(16, 32, 3).padding(1)),
        BatchNorm2d(32),
        Functional(torch::relu),
        MaxPool2d(MaxPool2dOptions(2).stride(2)),

        Flatten(),
        Linear(32 * 7 * 7, 10)
    );
}

// ============================================
// メイン：動作確認
// ============================================
int main() {
    std::cout << "LibTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    std::cout << "========================================" << std::endl;

    // ----- 1. MLPのテスト -----
    {
        auto model = std::make_shared<MLP>(784, 256, 10);
        model->to(torch::kCPU);  // または torch::kCUDA

        // ダミー入力: バッチサイズ4, 入力次元784
        auto input = torch::randn({4, 784});
        auto output = model->forward(input);

        std::cout << "\n[MLP]" << std::endl;
        std::cout << "  Input shape:  " << input.sizes() << std::endl;
        std::cout << "  Output shape: " << output.sizes() << std::endl;  // [4, 10]
        std::cout << "  Parameters:   " << model->parameters().size() << " tensors" << std::endl;
    }

    // ----- 2. CNNのテスト -----
    {
        auto model = std::make_shared<CNN>();
        auto input = torch::randn({2, 1, 28, 28});  // [N, C, H, W] = MNIST形式
        auto output = model->forward(input);

        std::cout << "\n[CNN]" << std::endl;
        std::cout << "  Input shape:  " << input.sizes() << std::endl;
        std::cout << "  Output shape: " << output.sizes() << std::endl;  // [2, 10]
    }

    // ----- 3. ResidualBlockのテスト -----
    {
        auto block = std::make_shared<ResidualBlock>(64, 64);
        auto input = torch::randn({2, 64, 32, 32});
        auto output = block->forward(input);

        std::cout << "\n[ResidualBlock]" << std::endl;
        std::cout << "  Input shape:  " << input.sizes() << std::endl;
        std::cout << "  Output shape: " << output.sizes() << std::endl;  // [2, 64, 32, 32]（同じ解像度）
    }

    // ----- 4. Sequentialのテスト -----
    {
        auto model = make_simple_cnn();
        auto input = torch::randn({8, 1, 28, 28});
        auto output = model->forward(input);

        std::cout << "\n[Sequential CNN]" << std::endl;
        std::cout << "  Input shape:  " << input.sizes() << std::endl;
        std::cout << "  Output shape: " << output.sizes() << std::endl;  // [8, 10]
    }

    // ----- 5. パラメータ数のカウント -----
    {
        auto model = std::make_shared<CNN>();
        int64_t total_params = 0;
        for (const auto& p : model->parameters()) {
            total_params += p.numel();
        }
        std::cout << "\n[CNN Total Parameters]: " << total_params << std::endl;
    }

    std::cout << "\nAll architecture tests passed!" << std::endl;
    return 0;
}