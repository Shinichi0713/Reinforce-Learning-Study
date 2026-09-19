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