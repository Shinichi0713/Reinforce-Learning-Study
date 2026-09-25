#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::RowVectorXf;

// --- 1. レイヤー基底クラス (Rustの Trait に相当) ---
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output_gradient, float lr) = 0;
};

// --- 2. 全結合層 (Dense Layer) ---
class Dense : public Layer {
private:
    Matrix weights;
    Vector biases;
    Matrix input_cache;

public:
    Dense(int in_features, int out_features) {
        // He初期化 (Kaiming Normal)
        std::random_device rd;
        std::mt19937 gen(rd());
        float std_dev = std::sqrt(2.0f / static_cast<float>(in_features));
        std::normal_distribution<float> d(0.0f, std_dev);

        weights = Matrix(in_features, out_features);
        for (int i = 0; i < in_features; ++i) {
            for (int j = 0; j < out_features; ++j) {
                weights(i, j) = d(gen);
            }
        }
        biases = Vector::Zero(out_features);
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        // Y = X * W + b (ブロードキャスト加算)
        return (input * weights).rowwise() + biases;
    }

    Matrix backward(const Matrix& output_gradient, float lr) override {
        // 勾配計算
        // dL/dW = X^T * dL/dY
        Matrix d_weights = input_cache.transpose() * output_gradient;
        // dL/db = sum(dL/dY, axis=0)
        Vector d_biases = output_gradient.colwise().sum();
        // dL/dX = dL/dY * W^T
        Matrix d_input = output_gradient * weights.transpose();

        // パラメータ更新 (SGD)
        weights -= d_weights * lr;
        biases -= d_biases * lr;

        return d_input;
    }
};

// --- 3. ReLU 活性化層 ---
class ReLU : public Layer {
private:
    Matrix output_cache;

public:
    Matrix forward(const Matrix& input) override {
        output_cache = input.unaryExpr([](float x) { return std::max(0.0f, x); });
        return output_cache;
    }

    Matrix backward(const Matrix& output_gradient, float lr) override {
        // ReLUの微分: 出力が > 0 なら 1.0, それ以外は 0.0
        Matrix d_relu = output_cache.unaryExpr([](float x) { return x > 0.0f ? 1.0f : 0.0f; });
        return output_gradient.cwiseProduct(d_relu);
    }
};

// --- 4. Sigmoid 活性化層 ---
class Sigmoid : public Layer {
private:
    Matrix output_cache;

public:
    Matrix forward(const Matrix& input) override {
        output_cache = input.unaryExpr([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
        return output_cache;
    }

    Matrix backward(const Matrix& output_gradient, float lr) override {
        // Sigmoidの微分: sig * (1 - sig)
        Matrix d_sigmoid = output_cache.binaryExpr(
            Matrix::Ones(output_cache.rows(), output_cache.cols()),
            [](float out, float one) { return out * (one - out); }
        );
        return output_gradient.cwiseProduct(d_sigmoid);
    }
};

// --- 5. 積層モデル (Sequential) ---
class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Matrix forward(Matrix x) {
        for (auto& layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }

    void backward(Matrix gradient, float lr) {
        // 逆伝播（後ろのレイヤーから順に処理）
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradient = (*it)->backward(gradient, lr);
        }
    }
};

// --- 6. メイン実行コード (XOR問題の学習) ---
int main() {
    // 2入力 -> 8隠れ層(ReLU) -> 4隠れ層(ReLU) -> 1出力(Sigmoid) のモデル構築
    Sequential model;
    model.add(std::make_unique<Dense>(2, 8));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(8, 4));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(4, 1));
    model.add(std::make_unique<Sigmoid>());

    // XOR問題のデータセット
    Matrix X(4, 2);
    X << 0.0f, 0.0f,
         0.0f, 1.0f,
         1.0f, 0.0f,
         1.0f, 1.0f;

    Matrix Y(4, 1);
    Y << 0.0f,
         1.0f,
         1.0f,
         0.0f;

    int epochs = 10000;
    float lr = 0.1f;

    std::cout << "--- 学習開始 ---" << std::endl;
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        // 1. 順伝播
        Matrix predictions = model.forward(X);

        // 2. 損失計算 (MSE)
        Matrix diff = predictions - Y;
        float loss = diff.array().square().mean();

        // 3. 損失の微分 (dL/dPred = 2 * (pred - y) / N)
        Matrix loss_gradient = 2.0f * diff / static_cast<float>(X.rows());

        // 4. 逆伝播による更新
        model.backward(loss_gradient, lr);

        if (epoch % 2000 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss << std::endl;
        }
    }

    std::cout << "\n--- 学習完了後の予測結果 ---" << std::endl;
    Matrix final_preds = model.forward(X);
    for (int i = 0; i < X.rows(); ++i) {
        std::cout << "Input: [" << X(i, 0) << ", " << X(i, 1) << "]"
                  << " -> Pred: " << final_preds(i, 0)
                  << " (Target: " << Y(i, 0) << ")" << std::endl;
    }

    return 0;
}