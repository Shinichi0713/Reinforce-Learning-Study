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