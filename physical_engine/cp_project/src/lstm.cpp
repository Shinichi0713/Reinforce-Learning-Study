#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

// シグモイド活性化関数
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

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

class LSTMCell {
public:
    int input_dim;
    int hidden_dim;

    // 4つの線形層 (x_t と h_{t-1} を結合したベクトルを入力とする)
    Linear linear_f; // Forget gate
    Linear linear_i; // Input gate
    Linear linear_c; // Candidate cell state
    Linear linear_o; // Output gate

    LSTMCell(int input_dim, int hidden_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim),
          linear_f(input_dim + hidden_dim, hidden_dim),
          linear_i(input_dim + hidden_dim, hidden_dim),
          linear_c(input_dim + hidden_dim, hidden_dim),
          linear_o(input_dim + hidden_dim, hidden_dim) {}

    // 1ステップの順伝播処理
    // input: [input_dim], h_prev: [hidden_dim], c_prev: [hidden_dim]
    // h_next: [hidden_dim], c_next: [hidden_dim]
    void forward(const float* input, const float* h_prev, const float* c_prev,
                 float* h_next, float* c_next) const {

        // 1. 入力 x_t と 前の隠れ状態 h_{t-1} を結合 [input_dim + hidden_dim]
        std::vector<float> concat_in(input_dim + hidden_dim);
        std::copy(input, input + input_dim, concat_in.begin());
        std::copy(h_prev, h_prev + hidden_dim, concat_in.begin() + input_dim);

        // ワークスペース領域の確保
        std::vector<float> f_out(hidden_dim);
        std::vector<float> i_out(hidden_dim);
        std::vector<float> c_tilde(hidden_dim);
        std::vector<float> o_out(hidden_dim);

        // 2. 各ゲートの線形計算と活性化関数
        linear_f.forward(concat_in.data(), f_out.data());
        linear_i.forward(concat_in.data(), i_out.data());
        linear_c.forward(concat_in.data(), c_tilde.data());
        linear_o.forward(concat_in.data(), o_out.data());

        for (int d = 0; d < hidden_dim; ++d) {
            float f_t = sigmoid(f_out[d]);
            float i_t = sigmoid(i_out[d]);
            float c_cand = tanh_act(c_tilde[d]);
            float o_t = sigmoid(o_out[d]);

            // 3. セル状態 C_t の更新
            c_next[d] = f_t * c_prev[d] + i_t * c_cand;

            // 4. 隠れ状態 h_t の更新
            h_next[d] = o_t * tanh_act(c_next[d]);
        }
    }
};

class LSTM {
public:
    LSTMCell cell;

    LSTM(int input_dim, int hidden_dim) : cell(input_dim, hidden_dim) {}

    // 入力系列 [seq_len x input_dim] を処理し、隠れ状態の系列 [seq_len x hidden_dim] を出力
    void forward(const std::vector<float>& input_seq, int seq_len,
                 std::vector<float>& output_seq) const {

        int input_dim = cell.input_dim;
        int hidden_dim = cell.hidden_dim;

        // 状態ベクトルの初期化 (h_0, c_0 はゼロベクトル)
        std::vector<float> h(hidden_dim, 0.0f);
        std::vector<float> c(hidden_dim, 0.0f);

        std::vector<float> h_next(hidden_dim);
        std::vector<float> c_next(hidden_dim);

        for (int t = 0; t < seq_len; ++t) {
            const float* x_t = input_seq.data() + t * input_dim;

            // 1ステップ実行
            cell.forward(x_t, h.data(), c.data(), h_next.data(), c_next.data());

            // 状態の更新
            h = h_next;
            c = c_next;

            // 出力系列に h_t を保存
            std::copy(h.begin(), h.end(), output_seq.begin() + t * hidden_dim);
        }
    }
};

int main() {
    // ハイパーパラメータ
    const int seq_len = 5;      // トークン数 / タイムステップ数
    const int input_dim = 4;    // 入力の特徴量次元
    const int hidden_dim = 8;   // 隠れ状態の次元数

    std::cout << "=== Building LSTM from Linear Layers in C++ ===" << std::endl;
    std::cout << "Sequence Length: " << seq_len << std::endl;
    std::cout << "Input Dimension: " << input_dim << std::endl;
    std::cout << "Hidden Dimension: " << hidden_dim << std::endl;

    // ダミーの入力データ生成 [seq_len x input_dim]
    std::vector<float> input_seq(seq_len * input_dim);
    std::mt19937 gen(1337);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (auto& val : input_seq) val = dis(gen);

    // LSTM モデルの構築と推論
    LSTM lstm(input_dim, hidden_dim);
    std::vector<float> output_seq(seq_len * hidden_dim, 0.0f);

    lstm.forward(input_seq, seq_len, output_seq);

    std::cout << "\n--- Forward Pass Completed ---" << std::endl;
    std::cout << "Output Hidden States h_t (First 2 Timesteps, First 4 dimensions):" << std::endl;
    for (int t = 0; t < 2; ++t) {
        std::cout << "t=" << t << ": ";
        for (int d = 0; d < 4; ++d) {
            std::cout << std::fixed << std::setprecision(4) << output_seq[t * hidden_dim + d] << " ";
        }
        std::cout << "..." << std::endl;
    }

    return 0;
}

