#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

// tanh 活性化関数
inline float tanh_act(float x) {
    return std::tanh(x);
}

// 線形層 (Linear Layer)
class Linear {
public:
    int in_features;
    int out_features;
    std::vector<float> weight; // Shape: [in_features x out_features]
    std::vector<float> bias;   // Shape: [out_features]

    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f),
                                 weight(in_f * out_f), bias(out_f) {
        std::mt19937 gen(42);
        // Xavier/Glorot 初期化
        float limit = std::sqrt(6.0f / (in_f + out_f));
        std::uniform_real_distribution<float> dis(-limit, limit);

        for (auto& w : weight) w = dis(gen);
        for (auto& b : bias) b = 0.0f;
    }

    // Input: [1 x in_features] -> Output: [1 x out_features]
    void forward(const float* input, float* output) const {
        for (int j = 0; j < out_features; ++j) {
            float sum = bias[j];
            for (int i = 0; i < in_features; ++i) {
                sum += input[i] * weight[i * out_features + j];
            }
            output[j] = sum;
        }
    }
};

class RNNCell {
public:
    int input_dim;
    int hidden_dim;

    // x_t [input_dim] と h_{t-1} [hidden_dim] を連結した入力（次元数: input_dim + hidden_dim）を受け取る線形層
    Linear linear_h;

    RNNCell(int input_dim, int hidden_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim),
          linear_h(input_dim + hidden_dim, hidden_dim) {}

    // 1ステップの順伝播処理
    // input: [input_dim], h_prev: [hidden_dim] -> h_next: [hidden_dim]
    void forward(const float* input, const float* h_prev, float* h_next) const {
        // 1. x_t と h_{t-1} を結合 [input_dim + hidden_dim]
        std::vector<float> concat_in(input_dim + hidden_dim);
        std::copy(input, input + input_dim, concat_in.begin());
        std::copy(h_prev, h_prev + hidden_dim, concat_in.begin() + input_dim);

        // 2. 線形計算 W * [x_t; h_{t-1}] + b
        std::vector<float> linear_out(hidden_dim);
        linear_h.forward(concat_in.data(), linear_out.data());

        // 3. tanh 活性化関数を適用して h_t を算出
        for (int d = 0; d < hidden_dim; ++d) {
            h_next[d] = tanh_act(linear_out[d]);
        }
    }
};

class RNN {
public:
    RNNCell cell;
    Linear out_proj; // 隠れ状態 h_t から最終出力 y_t への変換層

    RNN(int input_dim, int hidden_dim, int output_dim)
        : cell(input_dim, hidden_dim), out_proj(hidden_dim, output_dim) {}

    // 入力系列 [seq_len x input_dim] を処理し、出力系列 [seq_len x output_dim] を出力
    void forward(const std::vector<float>& input_seq, int seq_len,
                 std::vector<float>& output_seq) const {

        int input_dim = cell.input_dim;
        int hidden_dim = cell.hidden_dim;
        int output_dim = out_proj.out_features;

        // 隠れ状態 h_0 の初期化 (ゼロベクトル)
        std::vector<float> h(hidden_dim, 0.0f);
        std::vector<float> h_next(hidden_dim);

        for (int t = 0; t < seq_len; ++t) {
            const float* x_t = input_seq.data() + t * input_dim;

            // 1. RNNセルの順伝播 (h_t の更新)
            cell.forward(x_t, h.data(), h_next.data());
            h = h_next;

            // 2. 隠れ状態から最終出力 y_t への変換
            float* y_t = output_seq.data() + t * output_dim;
            out_proj.forward(h.data(), y_t);
        }
    }
};

int main() {
    // ハイパーパラメータ
    const int seq_len = 5;      // タイムステップ数 (系列長)
    const int input_dim = 4;    // 各ステップの入力次元数
    const int hidden_dim = 8;   // 隠れ状態の次元数
    const int output_dim = 2;   // 各ステップの出力次元数

    std::cout << "=== Building Simple RNN from Scratch in C++ ===" << std::endl;
    std::cout << "Sequence Length: " << seq_len << std::endl;
    std::cout << "Input Dimension: " << input_dim << std::endl;
    std::cout << "Hidden Dimension: " << hidden_dim << std::endl;
    std::cout << "Output Dimension: " << output_dim << std::endl;

    // ダミーの入力データ生成 [seq_len x input_dim]
    std::vector<float> input_seq(seq_len * input_dim);
    std::mt19937 gen(1337);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (auto& val : input_seq) val = dis(gen);

    // RNN モデル構築と推論の実行
    RNN rnn(input_dim, hidden_dim, output_dim);
    std::vector<float> output_seq(seq_len * output_dim, 0.0f);

    rnn.forward(input_seq, seq_len, output_seq);

    std::cout << "\n--- Forward Pass Completed ---" << std::endl;
    std::cout << "Output Predictions y_t for each Timestep:" << std::endl;
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "t=" << t << ": [ ";
        for (int d = 0; d < output_dim; ++d) {
            std::cout << std::fixed << std::setprecision(4) << output_seq[t * output_dim + d] << " ";
        }
        std::cout << "]" << std::endl;
    }

    return 0;
}

