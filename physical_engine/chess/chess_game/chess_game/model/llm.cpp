#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

// SiLU (Swish) 活性化関数
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// RMSNorm (Root Mean Square Layer Normalization)
class RMSNorm {
public:
    int dim;
    std::vector<float> weight; // スケールパラメータ gamma
    float eps;

    RMSNorm(int dim, float eps = 1e-6f) : dim(dim), weight(dim, 1.0f), eps(eps) {}

    void forward(const float* input, float* output, int N) const {
        for (int n = 0; n < N; ++n) {
            const float* in_ptr = input + n * dim;
            float* out_ptr = output + n * dim;

            // 二乗平均の計算
            float rms = 0.0f;
            for (int i = 0; i < dim; ++i) {
                rms += in_ptr[i] * in_ptr[i];
            }
            rms = std::sqrt((rms / dim) + eps);

            // 正規化 & スケーリング
            for (int i = 0; i < dim; ++i) {
                out_ptr[i] = (in_ptr[i] / rms) * weight[i];
            }
        }
    }
};

// 線形層 (Linear Layer)
class Linear {
public:
    int in_features;
    int out_features;
    std::vector<float> weight; // [in_features x out_features]

    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f), weight(in_f * out_f) {
        std::mt19937 gen(42);
        float limit = std::sqrt(6.0f / (in_f + out_f));
        std::uniform_real_distribution<float> dis(-limit, limit);
        for (auto& w : weight) w = dis(gen);
    }

    void forward(const float* input, float* output, int N) const {
        for (int n = 0; n < N; ++n) {
            for (int j = 0; j < out_features; ++j) {
                float sum = 0.0f; // Qwenの多くの線形層はバイアスなし
                for (int i = 0; i < in_features; ++i) {
                    sum += input[n * in_features + i] * weight[i * out_features + j];
                }
                output[n * out_features + j] = sum;
            }
        }
    }
};

// SwiGLU Feed-Forward Network
// FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
class SwiGLUFFN {
public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;

    SwiGLUFFN(int d_model, int hidden_dim)
        : gate_proj(d_model, hidden_dim),
          up_proj(d_model, hidden_dim),
          down_proj(hidden_dim, d_model) {}

    void forward(const float* input, float* output, int seq_len) const {
        int hidden_dim = gate_proj.out_features;
        std::vector<float> gate_out(seq_len * hidden_dim);
        std::vector<float> up_out(seq_len * hidden_dim);
        std::vector<float> activated(seq_len * hidden_dim);

        gate_proj.forward(input, gate_out.data(), seq_len);
        up_proj.forward(input, up_out.data(), seq_len);

        // SiLU(gate) * up
        for (size_t i = 0; i < gate_out.size(); ++i) {
            activated[i] = silu(gate_out[i]) * up_out[i];
        }

        down_proj.forward(activated.data(), output, seq_len);
    }
};