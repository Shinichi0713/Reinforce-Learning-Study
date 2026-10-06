#include <torch/torch.h>
#include <iostream>

// ----------------------------------------------------------------------
// 1. モデルの構築 (torch::nn::Module を継承)
// ----------------------------------------------------------------------
struct MLPImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    MLPImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim) {
        // レイヤーの登録 (register_module でパラメータを追跡可能にする)
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, output_dim));
    }

    // 順伝播 (Forward Pass)
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(MLP); // 参照カウント型のスマートポインタクラス (MLP) を自動生成

// ----------------------------------------------------------------------
// 2. メイン処理 (学習ループ)
// ----------------------------------------------------------------------
int main() {
    // CUDA (GPU) が利用可能か確認
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "Training on CPU." << std::endl;
    }

    // ハイパーパラメータの設定
    const int64_t input_dim = 10;
    const int64_t hidden_dim = 64;
    const int64_t output_dim = 2;
    const int64_t batch_size = 32;
    const int64_t num_epochs = 20;
    const double learning_rate = 0.001;

    // モデルのインスタンス化とデバイスへの移動
    MLP model(input_dim, hidden_dim, output_dim);
    model->to(device);

    // オプティマイザ (Adam) の設定
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));

    std::cout << "Starting training loop...\n";

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        model->train();

        // 1. ダミー入力データ [Batch Size, Input Dim] とターゲット [Batch Size]
        torch::Tensor inputs = torch::randn({batch_size, input_dim}, device);
        torch::Tensor targets = torch::randint(0, output_dim, {batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(device));

        // 2. 勾配のリセット
        optimizer.zero_grad();

        // 3. 順伝播 (Forward)
        torch::Tensor outputs = model->forward(inputs);

        // 4. 損失の計算 (Cross Entropy Loss)
        torch::Tensor loss = torch::nn::functional::cross_entropy(outputs, targets);

        // 5. 逆伝播 (Backward)
        loss.backward();

        // 6. パラメータの更新
        optimizer.step();

        // エポックごとの進捗表示
        if (epoch % 5 == 0 || epoch == 1) {
            std::cout << "Epoch [" << epoch << "/" << num_epochs << "] "
                      << "| Loss: " << loss.item<double>() << std::endl;
        }
    }

    // モデル重みの保存 (.pt 形式)
    torch::save(model, "mlp_model.pt");
    std::cout << "Model successfully saved to mlp_model.pt\n";

    return 0;
}