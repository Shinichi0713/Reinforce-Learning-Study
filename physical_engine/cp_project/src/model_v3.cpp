#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <immintrin.h> // AVX/AVX2
#include <omp.h>       // OpenMP

class DenseLayerOptimized {
public:
    int in_features;
    int out_features;

    // 1次元配列によるフラット化 (Row-major配置)
    std::vector<float> weights; // Size: in_features * out_features
    std::vector<float> bias;    // Size: out_features

    DenseLayerOptimized(int in_f, int out_f)
        : in_features(in_f), out_features(out_f),
          weights(in_f * out_f), bias(out_f) {
        
        std::mt19937 gen(1337);
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

        for (auto& w : weights) w = dis(gen);
        for (auto& b : bias) b = dis(gen);
    }

    // 順伝播: Output = Input * Weights + Bias
    // Input:  [N x in_features]
    // Output: [N x out_features]
    void forward(const float* __restrict input, float* __restrict output, int batch_size) const {
        
        // OpenMPによるバッチ・出力行の並列化
        #pragma omp parallel for collapse(2) schedule(static)
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < out_features; ++j) {
                
                // バイアス値で初期化
                float sum = bias[j];
                
                // AVX-256 (8 float並列) アキュムレータ
                __m256 vsum = _mm256_setzero_ps();

                int i = 0;
                // SIMD命令によるループアンロール (8要素ずつ一括計算)
                for (; i <= in_features - 8; i += 8) {
                    // Input[b, i..i+7] をロード
                    __m256 vin = _mm256_loadu_ps(&input[b * in_features + i]);
                    
                    // 重み行は連続アクセスできない(転置なしの場合)ため、ギャザーまたは要素ロード
                    // キャッシュ効率向上のため重みアクセスをロード
                    alignas(32) float w_tmp[8];
                    for (int k = 0; k < 8; ++k) {
                        w_tmp[k] = weights[(i + k) * out_features + j];
                    }
                    __m256 vw = _mm256_load_ps(w_tmp);

                    // 積加算 (FMA)
                    vsum = _mm256_fmadd_ps(vin, vw, vsum);
                }

                // AVXレジスタ内の8要素を足し合わせる
                alignas(32) float res[8];
                _mm256_store_ps(res, vsum);
                for (int k = 0; k < 8; ++k) {
                    sum += res[k];
                }

                // 端数処理 (8の倍数からあふれた要素)
                for (; i < in_features; ++i) {
                    sum += input[b * in_features + i] * weights[i * out_features + j];
                }

                // ReLU活性化関数をそのまま適用
                output[b * out_features + j] = sum > 0.0f ? sum : 0.0f;
            }
        }
    }
};

int main() {
    const int batch_size = 128;
    const int in_features = 1024;
    const int out_features = 2048;

    DenseLayerOptimized layer(in_features, out_features);

    // 1次元にフラット化された入力・出力バッファ
    std::vector<float> input(batch_size * in_features, 1.0f);
    std::vector<float> output(batch_size * out_features, 0.0f);

    std::cout << "Threads: " << omp_get_max_threads() << std::endl;

    // 処理速度計測
    auto start = std::chrono::high_resolution_clock::now();
    
    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        layer.forward(input.data(), output.data(), batch_size);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Avg Forward Pass Time: " << elapsed.count() / iterations << " ms" << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstdint>

// -------------------------------------------------------------
// 1. 量子化ブロックのデータ構造定義
// -------------------------------------------------------------

// Q8_0: 32要素ごとに 1個の FP32 スケール因子(d) + 32個の INT8 重み(qs)
struct BlockQ8_0 {
    float d;            // スケール因子 (Scale Factor)
    int8_t qs[32];      // 量子化された重み (8-bit)
};

// Q4_0: 32要素ごとに 1個の FP32 スケール因子(d) + 16個の UINT8 (ニブル分割で4-bit×32個分)
struct BlockQ4_0 {
    float d;            // スケール因子
    uint8_t qs[16];     // 量子化された重み (下位4bit: 前半16個, 上位4bit: 後半16個)
};

// -------------------------------------------------------------
// 2. 量子化（Quantize）関数
// -------------------------------------------------------------

// FP32 配列 -> Q8_0 ブロック配列
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int k) {
    const int nb = k / 32; // ブロック数 (k は 32 の倍数であること)

    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = std::max(amax, std::abs(x[i * 32 + j]));
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = d;
        for (int j = 0; j < 32; ++j) {
            const float val = x[i * 32 + j] * id;
            y[i].qs[j] = static_cast<int8_t>(std::round(val));
        }
    }
}

// FP32 配列 -> Q4_0 ブロック配列
void quantize_row_q4_0(const float* x, BlockQ4_0* y, int k) {
    const int nb = k / 32;

    for (int i = 0; i < nb; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            amax = std::max(amax, std::abs(x[i * 32 + j]));
        }

        const float d = amax / 7.0f; // -8 ~ +7 範囲へマッピング
        const float id = d ? 1.0f / d : 0.0f;

        y[i].d = d;
        for (int j = 0; j < 16; ++j) {
            // 前半 16 個 (下位 4 ビット)
            const float x0 = x[i * 32 + j] * id;
            const uint8_t q0 = static_cast<uint8_t>(std::clamped(static_cast<int>(std::round(x0)) + 8, 0, 15));

            // 後半 16 個 (上位 4 ビット)
            const float x1 = x[i * 32 + j + 16] * id;
            const uint8_t q1 = static_cast<uint8_t>(std::clamped(static_cast<int>(std::round(x1)) + 8, 0, 15));

            y[i].qs[j] = q0 | (q1 << 4);
        }
    }
}

// -------------------------------------------------------------
// 3. 量子化重みを用いた 行列積（GEMV）演算カーネル
// -------------------------------------------------------------

// Q8_0 重みを用いた 順伝播 (Vector-Matrix Multiplication)
// y = x * W
void gemv_q8_0(const float* x, const BlockQ8_0* W, float* y, int in_f, int out_f) {
    const int nb = in_f / 32;

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out_f; ++j) {
        float sum = 0.0f;

        for (int b = 0; b < nb; ++b) {
            const BlockQ8_0& block = W[j * nb + b];
            const float d = block.d;

            float block_sum = 0.0f;
            for (int i = 0; i < 32; ++i) {
                block_sum += x[b * 32 + i] * static_cast<float>(block.qs[i]);
            }
            sum += block_sum * d; // 最後にスケール因子を積算
        }
        y[j] = sum;
    }
}

// Q4_0 重みを用いた 順伝播
void gemv_q4_0(const float* x, const BlockQ4_0* W, float* y, int in_f, int out_f) {
    const int nb = in_f / 32;

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < out_f; ++j) {
        float sum = 0.0f;

        for (int b = 0; b < nb; ++b) {
            const BlockQ4_0& block = W[j * nb + b];
            const float d = block.d;

            float block_sum = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const uint8_t packed = block.qs[i];
                const int q0 = static_cast<int>(packed & 0x0F) - 8;
                const int q1 = static_cast<int>(packed >> 4) - 8;

                block_sum += x[b * 32 + i] * static_cast<float>(q0);
                block_sum += x[b * 32 + i + 16] * static_cast<float>(q1);
            }
            sum += block_sum * d;
        }
        y[j] = sum;
    }
}

// -------------------------------------------------------------
// 4. 検証およびベンチマーク
// -------------------------------------------------------------
int main() {
    const int in_features = 4096;
    const int out_features = 4096;

    std::cout << "=== Quantization (Q8_0 / Q4_0) Benchmark ===" << std::endl;
    std::cout << "Matrix Size: " << in_features << " x " << out_features << std::endl;

    // 元の重み (FP32)
    std::vector<float> h_weights_fp32(in_features * out_features);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& w : h_weights_fp32) w = dis(gen);

    // 入力ベクトル
    std::vector<float> input(in_features, 1.0f);
    std::vector<float> output_q8(out_features, 0.0f);
    std::vector<float> output_q4(out_features, 0.0f);

    // 重みの量子化
    const int num_blocks_per_row = in_features / 32;
    std::vector<BlockQ8_0> weights_q8(out_features * num_blocks_per_row);
    std::vector<BlockQ4_0> weights_q4(out_features * num_blocks_per_row);

    for (int j = 0; j < out_features; ++j) {
        quantize_row_q8_0(&h_weights_fp32[j * in_features], &weights_q8[j * num_blocks_per_row], in_features);
        quantize_row_q4_0(&h_weights_fp32[j * in_features], &weights_q4[j * num_blocks_per_row], in_features);
    }

    // メモリ消費量の比較
    const size_t fp32_size = h_weights_fp32.size() * sizeof(float);
    const size_t q8_size = weights_q8.size() * sizeof(BlockQ8_0);
    const size_t q4_size = weights_q4.size() * sizeof(BlockQ4_0);

    std::cout << "\n--- Memory Footprint ---" << std::endl;
    std::cout << "FP32 Size: " << fp32_size / (1024.0 * 1024.0) << " MB (100%)" << std::endl;
    std::cout << "Q8_0 Size: " << q8_size / (1024.0 * 1024.0) << " MB (" 
              << (q8_size * 100.0 / fp32_size) << "%)" << std::endl;
    std::cout << "Q4_0 Size: " << q4_size / (1024.0 * 1024.0) << " MB (" 
              << (q4_size * 100.0 / fp32_size) << "%)" << std::endl;

    // 推論速度の計測
    auto start_q8 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        gemv_q8_0(input.data(), weights_q8.data(), output_q8.data(), in_features, out_features);
    }
    auto end_q8 = std::chrono::high_resolution_clock::now();

    auto start_q4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        gemv_q4_0(input.data(), weights_q4.data(), output_q4.data(), in_features, out_features);
    }
    auto end_q4 = std::chrono::high_resolution_clock::now();

    std::cout << "\n--- Performance ---" << std::endl;
    std::cout << "Q8_0 Avg Time: " << std::chrono::duration<double, std::milli>(end_q8 - start_q8).count() / 50.0 << " ms" << std::endl;
    std::cout << "Q4_0 Avg Time: " << std::chrono::duration<double, std::milli>(end_q4 - start_q4).count() / 50.0 << " ms" << std::endl;

    return 0;
}


#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>

// GELU 活性化関数 (近似式)
inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))));
}

// Softmax (数値安定化版: max引き)
void softmax(float* x, int size) {
    float max_val = *std::max_element(x, x + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

// 線形層 (Linear Layer / Fully Connected Layer)
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

    // Input: [N x in_features] -> Output: [N x out_features]
    void forward(const float* input, float* output, int N) const {
        for (int n = 0; n < N; ++n) {
            for (int j = 0; j < out_features; ++j) {
                float sum = bias[j];
                for (int i = 0; i < in_features; ++i) {
                    sum += input[n * in_features + i] * weight[i * out_features + j];
                }
                output[n * out_features + j] = sum;
            }
        }
    }
};

// Layer Normalization
class LayerNorm {
public:
    int dim;
    std::vector<float> gamma;
    std::vector<float> beta;
    float eps;

    LayerNorm(int dim, float eps = 1e-5f) : dim(dim), gamma(dim, 1.0f), beta(dim, 0.0f), eps(eps) {}

    // Input/Output: [N x dim]
    void forward(const float* input, float* output, int N) const {
        for (int n = 0; n < N; ++n) {
            const float* in_ptr = input + n * dim;
            float* out_ptr = output + n * dim;

            // 平均の計算
            float mean = 0.0f;
            for (int i = 0; i < dim; ++i) mean += in_ptr[i];
            mean /= dim;

            // 分散の計算
            float var = 0.0f;
            for (int i = 0; i < dim; ++i) {
                float diff = in_ptr[i] - mean;
                var += diff * diff;
            }
            var /= dim;

            // 正規化 & アフィン変換
            float inv_std = 1.0f / std::sqrt(var + eps);
            for (int i = 0; i < dim; ++i) {
                out_ptr[i] = gamma[i] * ((in_ptr[i] - mean) * inv_std) + beta[i];
            }
        }
    }
};

// Feed-Forward Network (FFN)
// Linear1 -> GELU -> Linear2
class FeedForward {
public:
    Linear w1;
    Linear w2;

    FeedForward(int d_model, int d_ff) : w1(d_model, d_ff), w2(d_ff, d_model) {}

    void forward(const float* input, float* output, int seq_len) const {
        std::vector<float> hidden(seq_len * w1.out_features);
        
        // 1. w1: [seq_len x d_model] -> [seq_len x d_ff]
        w1.forward(input, hidden.data(), seq_len);

        // 2. GELU 活性化
        for (auto& val : hidden) val = gelu(val);

        // 3. w2: [seq_len x d_ff] -> [seq_len x d_model]
        w2.forward(hidden.data(), output, seq_len);
    }
};

class MultiHeadAttention {
public:
    int d_model;
    int num_heads;
    int head_dim;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    MultiHeadAttention(int d_model, int num_heads)
        : d_model(d_model), num_heads(num_heads), head_dim(d_model / num_heads),
          q_proj(d_model, d_model), k_proj(d_model, d_model),
          v_proj(d_model, d_model), out_proj(d_model, d_model) {}

    void forward(const float* input, float* output, int seq_len) const {
        std::vector<float> Q(seq_len * d_model);
        std::vector<float> K(seq_len * d_model);
        std::vector<float> V(seq_len * d_model);

        // Q, K, V プロジェクション
        q_proj.forward(input, Q.data(), seq_len);
        k_proj.forward(input, K.data(), seq_len);
        v_proj.forward(input, V.data(), seq_len);

        std::vector<float> concat_attn_out(seq_len * d_model, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // ヘッドごとの並行処理
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) { // Query Token
                std::vector<float> attn_scores(seq_len, 0.0f);

                // 1. Q * K^T / sqrt(d_k)
                for (int j = 0; j < seq_len; ++j) { // Key Token
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        float q_val = Q[i * d_model + h * head_dim + d];
                        float k_val = K[j * d_model + h * head_dim + d];
                        score += q_val * k_val;
                    }
                    attn_scores[j] = score * scale;
                }

                // 2. Softmax(Scores)
                softmax(attn_scores.data(), seq_len);

                // 3. Attention Weight * V
                for (int d = 0; d < head_dim; ++d) {
                    float head_out = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        float v_val = V[j * d_model + h * head_dim + d];
                        head_out += attn_scores[j] * v_val;
                    }
                    concat_attn_out[i * d_model + h * head_dim + d] = head_out;
                }
            }
        }

        // 最終出力のプロジェクション
        out_proj.forward(concat_attn_out.data(), output, seq_len);
    }
};

class TransformerBlock {
public:
    MultiHeadAttention mha;
    LayerNorm norm1;
    FeedForward ffn;
    LayerNorm norm2;

    TransformerBlock(int d_model, int num_heads, int d_ff)
        : mha(d_model, num_heads), norm1(d_model), ffn(d_model, d_ff), norm2(d_model) {}

    void forward(const float* input, float* output, int seq_len) const {
        int total_size = seq_len * mha.d_model;

        // --- 1. Multi-Head Attention Sub-layer ---
        std::vector<float> attn_out(total_size);
        mha.forward(input, attn_out.data(), seq_len);

        // 残差接続 + LayerNorm (Pre-LN 構造)
        std::vector<float> residual1(total_size);
        for (int i = 0; i < total_size; ++i) residual1[i] = input[i] + attn_out[i];
        
        std::vector<float> norm1_out(total_size);
        norm1.forward(residual1.data(), norm1_out.data(), seq_len);

        // --- 2. Feed-Forward Sub-layer ---
        std::vector<float> ffn_out(total_size);
        ffn.forward(norm1_out.data(), ffn_out.data(), seq_len);

        // 残差接続 + LayerNorm
        std::vector<float> residual2(total_size);
        for (int i = 0; i < total_size; ++i) residual2[i] = norm1_out[i] + ffn_out[i];

        norm2.forward(residual2.data(), output, seq_len);
    }
};

int main() {
    // ハイパーパラメータ
    const int seq_len = 4;     // トークン列長 (例: "I", "am", "a", "robot")
    const int d_model = 16;    // 隠れ層の次元数
    const int num_heads = 4;   // アテンションヘッド数
    const int d_ff = 64;       // FFN中間層の次元数

    std::cout << "=== Building Transformer Block in C++ ===" << std::endl;
    std::cout << "Sequence Length: " << seq_len << std::endl;
    std::cout << "d_model:         " << d_model << std::endl;
    std::cout << "Heads:           " << num_heads << std::endl;

    // 入力テンソルの作成 [seq_len x d_model]
    std::vector<float> x(seq_len * d_model);
    std::mt19937 gen(1337);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (auto& val : x) val = dis(gen);

    // Transformer ブロック構築・順伝播実行
    TransformerBlock block(d_model, num_heads, d_ff);
    std::vector<float> out(seq_len * d_model, 0.0f);

    block.forward(x.data(), out.data(), seq_len);

    std::cout << "\n--- Forward Pass Completed ---" << std::endl;
    std::cout << "Output Tensor (First 2 Tokens, First 4 dimensions):" << std::endl;
    for (int t = 0; t < 2; ++t) {
        std::cout << "Token [" << t << "]: ";
        for (int d = 0; d < 4; ++d) {
            std::cout << std::fixed << std::setprecision(4) << out[t * d_model + d] << " ";
        }
        std::cout << "..." << std::endl;
    }

    return 0;
}