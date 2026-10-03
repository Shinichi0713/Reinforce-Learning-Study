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

// RMSNorm
class RMSNorm {
public:
    int dim;
    std::vector<float> weight;
    float eps;

    RMSNorm(int dim, float eps = 1e-6f) : dim(dim), weight(dim, 1.0f), eps(eps) {}

    void forward(const float* input, float* output, int N) const {
        for (int n = 0; n < N; ++n) {
            const float* in_ptr = input + n * dim;
            float* out_ptr = output + n * dim;

            float rms = 0.0f;
            for (int i = 0; i < dim; ++i) rms += in_ptr[i] * in_ptr[i];
            rms = std::sqrt((rms / dim) + eps);

            for (int i = 0; i < dim; ++i) {
                out_ptr[i] = (in_ptr[i] / rms) * weight[i];
            }
        }
    }
};

// 線形層
class Linear {
public:
    int in_features;
    int out_features;
    std::vector<float> weight;

    Linear(int in_f, int out_f) : in_features(in_f), out_features(out_f), weight(in_f * out_f) {
        std::mt19937 gen(42);
        float limit = std::sqrt(6.0f / (in_f + out_f));
        std::uniform_real_distribution<float> dis(-limit, limit);
        for (auto& w : weight) w = dis(gen);
    }

    void forward(const float* input, float* output, int N) const {
        for (int n = 0; n < N; ++n) {
            for (int j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < in_features; ++i) {
                    sum += input[n * in_features + i] * weight[i * out_features + j];
                }
                output[n * out_features + j] = sum;
            }
        }
    }
};

// SwiGLU FFN
class SwiGLUFFN {
public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;

    SwiGLUFFN(int d_model, int hidden_dim)
        : gate_proj(d_model, hidden_dim), up_proj(d_model, hidden_dim), down_proj(hidden_dim, d_model) {}

    void forward(const float* input, float* output, int seq_len) const {
        int hidden_dim = gate_proj.out_features;
        std::vector<float> gate_out(seq_len * hidden_dim);
        std::vector<float> up_out(seq_len * hidden_dim);
        std::vector<float> activated(seq_len * hidden_dim);

        gate_proj.forward(input, gate_out.data(), seq_len);
        up_proj.forward(input, up_out.data(), seq_len);

        for (size_t i = 0; i < gate_out.size(); ++i) {
            activated[i] = silu(gate_out[i]) * up_out[i];
        }

        down_proj.forward(activated.data(), output, seq_len);
    }
};

// --- KV Cache 構造体 ---
struct KVCache {
    // 形状: [cached_seq_len x num_kv_heads x head_dim]
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    int cached_seq_len = 0;

    void clear() {
        k_cache.clear();
        v_cache.clear();
        cached_seq_len = 0;
    }

    // 新しい Key/Value ベクトルをキャッシュ末尾に追加
    void append(const float* new_k, const float* new_v, int new_tokens, int num_kv_heads, int head_dim) {
        int add_elements = new_tokens * num_kv_heads * head_dim;
        k_cache.insert(k_cache.end(), new_k, new_k + add_elements);
        v_cache.insert(v_cache.end(), new_v, new_v + add_elements);
        cached_seq_len += new_tokens;
    }
};