#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>
#include <numeric>

// ============================================================================
// 1. 高精度数値演算 & 安定化アテンション基本関数
// ============================================================================

// SiLU (Swish) 活性化関数: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// 数値的に安定した Softmax（log-sum-exp トリックによるアンダーフロー防止）
void safe_softmax_inplace(float* x, int size) {
    float max_val = *std::max_element(x, x + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// 高精度 Row-Major 行列積演算 Y = X * W
void matmul_cpu_high_precision(const float* X, const float* W, float* Y, int N, int in_f, int out_f) {
    for (int n = 0; n < N; ++n) {
        const float* x_row = X + n * in_f;
        float* y_row = Y + n * out_f;
        for (int j = 0; j < out_f; ++j) {
            // アキュムレータを double にすることで精度劣化を防ぐ
            double sum = 0.0;
            for (int i = 0; i < in_f; ++i) {
                sum += static_cast<double>(x_row[i]) * static_cast<double>(W[i * out_f + j]);
            }
            y_row[j] = static_cast<float>(sum);
        }
    }
}

// ============================================================================
// 2. 高精度ニューラルネットワーク基本層
// ============================================================================

// 線形層 (Linear Layer)
class Linear {
public:
    int in_features;
    int out_features;
    std::vector<float> weight; // [in_features x out_features]

    Linear(int in_f, int out_f, std::mt19937& gen)
        : in_features(in_f), out_features(out_f), weight(in_f * out_f) {
        // Kaiming / He 標準正規分布初期化 (精度向上用)
        float stddev = std::sqrt(2.0f / in_f);
        std::normal_distribution<float> dis(0.0f, stddev);
        for (auto& w : weight) w = dis(gen);
    }

    void forward(const float* input, float* output, int N) const {
        matmul_cpu_high_precision(input, weight.data(), output, N, in_features, out_features);
    }
};

// RMSNorm (Root Mean Square Normalization)
class RMSNorm {
public:
    int dim;
    std::vector<float> weight; // Gamma パラメータ
    float eps;

    RMSNorm(int dim, float eps = 1e-6f) : dim(dim), weight(dim, 1.0f), eps(eps) {}

    void forward(const float* input, float* output, int N) const {
        for (int n = 0; n < N; ++n) {
            const float* in_ptr = input + n * dim;
            float* out_ptr = output + n * dim;

            double square_sum = 0.0;
            for (int i = 0; i < dim; ++i) {
                square_sum += static_cast<double>(in_ptr[i]) * static_cast<double>(in_ptr[i]);
            }
            float rms = static_cast<float>(std::sqrt((square_sum / dim) + static_cast<double>(eps)));
            float inv_rms = 1.0f / rms;

            for (int i = 0; i < dim; ++i) {
                out_ptr[i] = (in_ptr[i] * inv_rms) * weight[i];
            }
        }
    }
};

// SwiGLU Feed-Forward Network
class SwiGLUFFN {
public:
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;

    SwiGLUFFN(int d_model, int hidden_dim, std::mt19937& gen)
        : gate_proj(d_model, hidden_dim, gen),
          up_proj(d_model, hidden_dim, gen),
          down_proj(hidden_dim, d_model, gen) {}

    void forward(const float* input, float* output, int seq_len) const {
        int hidden_dim = gate_proj.out_features;
        std::vector<float> gate_out(seq_len * hidden_dim);
        std::vector<float> up_out(seq_len * hidden_dim);
        std::vector<float> act_out(seq_len * hidden_dim);

        gate_proj.forward(input, gate_out.data(), seq_len);
        up_proj.forward(input, up_out.data(), seq_len);

        for (size_t i = 0; i < gate_out.size(); ++i) {
            act_out[i] = silu(gate_out[i]) * up_out[i];
        }

        down_proj.forward(act_out.data(), output, seq_len);
    }
};

// ============================================================================
// 3. KV キャッシュ構造体
// ============================================================================

struct KVCache {
    std::vector<float> k_cache; // [cached_seq_len x num_kv_heads x head_dim]
    std::vector<float> v_cache; // [cached_seq_len x num_kv_heads x head_dim]
    int cached_seq_len = 0;

    void reset() {
        k_cache.clear();
        v_cache.clear();
        cached_seq_len = 0;
    }

    void append(const float* new_k, const float* new_v, int num_tokens, int num_kv_heads, int head_dim) {
        int elements = num_tokens * num_kv_heads * head_dim;
        k_cache.insert(k_cache.end(), new_k, new_k + elements);
        v_cache.insert(v_cache.end(), new_v, new_v + elements);
        cached_seq_len += num_tokens;
    }
};

// ============================================================================
// 4. アテンション層 (GQA + 高精度 double RoPE)
// ============================================================================

class QwenAttention {
public:
    int d_model;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int num_queries_per_kv;
    float rope_theta;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    QwenAttention(int d_model, int num_heads, int num_kv_heads, std::mt19937& gen, float rope_theta = 1000000.0f)
        : d_model(d_model), num_heads(num_heads), num_kv_heads(num_kv_heads),
          head_dim(d_model / num_heads),
          num_queries_per_kv(num_heads / num_kv_heads),
          rope_theta(rope_theta),
          q_proj(d_model, num_heads * head_dim, gen),
          k_proj(d_model, num_kv_heads * head_dim, gen),
          v_proj(d_model, num_kv_heads * head_dim, gen),
          out_proj(num_heads * head_dim, d_model, gen) {}

    // double 精度による倍精度 RoPE 回転の計算
    void apply_rope_high_precision(float* vec, int abs_pos, int dim) const {
        for (int i = 0; i < dim; i += 2) {
            double freq = 1.0 / std::pow(static_cast<double>(rope_theta), static_cast<double>(i) / dim);
            double theta = static_cast<double>(abs_pos) * freq;
            double cos_t = std::cos(theta);
            double sin_t = std::sin(theta);

            double v0 = static_cast<double>(vec[i]);
            double v1 = static_cast<double>(vec[i + 1]);

            vec[i]     = static_cast<float>(v0 * cos_t - v1 * sin_t);
            vec[i + 1] = static_cast<float>(v0 * sin_t + v1 * cos_t);
        }
    }

    void forward(const float* input, float* output, int num_tokens, KVCache& kv_cache) const {
        int start_pos = kv_cache.cached_seq_len;

        std::vector<float> Q(num_tokens * num_heads * head_dim);
        std::vector<float> new_K(num_tokens * num_kv_heads * head_dim);
        std::vector<float> new_V(num_tokens * num_kv_heads * head_dim);

        q_proj.forward(input, Q.data(), num_tokens);
        k_proj.forward(input, new_K.data(), num_tokens);
        v_proj.forward(input, new_V.data(), num_tokens);

        for (int t = 0; t < num_tokens; ++t) {
            int abs_pos = start_pos + t;
            for (int h = 0; h < num_heads; ++h) {
                apply_rope_high_precision(Q.data() + (t * num_heads + h) * head_dim, abs_pos, head_dim);
            }
            for (int h = 0; h < num_kv_heads; ++h) {
                apply_rope_high_precision(new_K.data() + (t * num_kv_heads + h) * head_dim, abs_pos, head_dim);
            }
        }

        kv_cache.append(new_K.data(), new_V.data(), num_tokens, num_kv_heads, head_dim);

        int total_seq_len = kv_cache.cached_seq_len;
        const float* K_full = kv_cache.k_cache.data();
        const float* V_full = kv_cache.v_cache.data();

        std::vector<float> concat_attn_out(num_tokens * num_heads * head_dim, 0.0f);
        float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));

        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / num_queries_per_kv;

            for (int i = 0; i < num_tokens; ++i) {
                int current_abs_pos = start_pos + i;
                std::vector<float> attn_scores(total_seq_len, -1e9f);

                for (int j = 0; j <= current_abs_pos; ++j) {
                    double score = 0.0;
                    for (int d = 0; d < head_dim; ++d) {
                        float q_val = Q[(i * num_heads + h) * head_dim + d];
                        float k_val = K_full[(j * num_kv_heads + kv_h) * head_dim + d];
                        score += static_cast<double>(q_val) * static_cast<double>(k_val);
                    }
                    attn_scores[j] = static_cast<float>(score) * scale;
                }

                safe_softmax_inplace(attn_scores.data(), current_abs_pos + 1);

                for (int d = 0; d < head_dim; ++d) {
                    double head_out = 0.0;
                    for (int j = 0; j <= current_abs_pos; ++j) {
                        float v_val = V_full[(j * num_kv_heads + kv_h) * head_dim + d];
                        head_out += static_cast<double>(attn_scores[j]) * static_cast<double>(v_val);
                    }
                    concat_attn_out[(i * num_heads + h) * head_dim + d] = static_cast<float>(head_out);
                }
            }
        }

        out_proj.forward(concat_attn_out.data(), output, num_tokens);
    }
};

// ============================================================================
// 5. Transformer Block & LLM アーキテクチャ
// ============================================================================

class QwenBlock {
public:
    QwenAttention attn;
    RMSNorm norm1;
    SwiGLUFFN ffn;
    RMSNorm norm2;

    QwenBlock(int d_model, int num_heads, int num_kv_heads, int intermediate_size, std::mt19937& gen, float rope_theta = 1000000.0f)
        : attn(d_model, num_heads, num_kv_heads, gen, rope_theta),
          norm1(d_model),
          ffn(d_model, intermediate_size, gen),
          norm2(d_model) {}

    void forward(const float* input, float* output, int num_tokens, KVCache& kv_cache) const {
        int d_model = attn.d_model;
        int total_size = num_tokens * d_model;

        std::vector<float> norm1_out(total_size);
        norm1.forward(input, norm1_out.data(), num_tokens);

        std::vector<float> attn_out(total_size);
        attn.forward(norm1_out.data(), attn_out.data(), num_tokens, kv_cache);

        std::vector<float> residual1(total_size);
        for (int i = 0; i < total_size; ++i) residual1[i] = input[i] + attn_out[i];

        std::vector<float> norm2_out(total_size);
        norm2.forward(residual1.data(), norm2_out.data(), num_tokens);

        std::vector<float> ffn_out(total_size);
        ffn.forward(norm2_out.data(), ffn_out.data(), num_tokens);

        for (int i = 0; i < total_size; ++i) output[i] = residual1[i] + ffn_out[i];
    }
};

class QwenLLM {
public:
    int vocab_size;
    int d_model;
    int num_layers;

    std::vector<float> token_embedding_table;
    std::vector<QwenBlock> layers;
    RMSNorm final_norm;
    Linear lm_head;

    std::vector<KVCache> layer_caches;
    std::mt19937 rng;

    QwenLLM(int vocab_size, int d_model, int num_layers, int num_heads, int num_kv_heads, int intermediate_size, uint32_t seed = 42)
        : vocab_size(vocab_size), d_model(d_model), num_layers(num_layers),
          token_embedding_table(vocab_size * d_model),
          final_norm(d_model),
          lm_head(d_model, vocab_size, std::mt19937(seed + 999)),
          layer_caches(num_layers),
          rng(seed) {

        std::mt19937 gen(seed);
        float stddev = std::sqrt(1.0f / d_model);
        std::normal_distribution<float> dis(0.0f, stddev);

        for (auto& e : token_embedding_table) e = dis(gen);

        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(d_model, num_heads, num_kv_heads, intermediate_size, gen);
        }
    }

    void reset_cache() {
        for (auto& cache : layer_caches) cache.reset();
    }

    void forward(const std::vector<int>& input_ids, std::vector<float>& logits) {
        int num_tokens = static_cast<int>(input_ids.size());

        std::vector<float> hidden_states(num_tokens * d_model);
        for (int t = 0; t < num_tokens; ++t) {
            int token_id = input_ids[t];
            const float* emb_ptr = token_embedding_table.data() + token_id * d_model;
            std::copy(emb_ptr, emb_ptr + d_model, hidden_states.begin() + t * d_model);
        }

        std::vector<float> layer_output(num_tokens * d_model);
        for (int l = 0; l < num_layers; ++l) {
            layers[l].forward(hidden_states.data(), layer_output.data(), num_tokens, layer_caches[l]);
            hidden_states = layer_output;
        }

        std::vector<float> norm_output(num_tokens * d_model);
        final_norm.forward(hidden_states.data(), norm_output.data(), num_tokens);

        logits.resize(num_tokens * vocab_size);
        lm_head.forward(norm_output.data(), logits.data(), num_tokens);
    }

    // Top-K & Top-P (Nucleus) サンプリングアルゴリズム
    int sample_advanced(const float* logits_ptr, float temperature = 0.7f, int top_k = 40, float top_p = 0.9f) {
        if (temperature <= 0.0f) {
            // Greedy Search (Argmax)
            return static_cast<int>(std::distance(logits_ptr, std::max_element(logits_ptr, logits_ptr + vocab_size)));
        }

        struct TokenProb {
            int id;
            float prob;
        };

        std::vector<float> temp_logits(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            temp_logits[v] = logits_ptr[v] / temperature;
        }
        safe_softmax_inplace(temp_logits.data(), vocab_size);

        std::vector<TokenProb> candidates(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            candidates[i] = {i, temp_logits[i]};
        }

        // 確率降順にソート
        std::sort(candidates.begin(), candidates.end(), [](const TokenProb& a, const TokenProb& b) {
            return a.prob > b.prob;
        });

        // 1. Top-K フィルタリング
        if (top_k > 0 && top_k < vocab_size) {
            candidates.resize(top_k);
        }

        // 2. Top-P (Nucleus) フィルタリング
        float cum_sum = 0.0f;
        int cutoff_index = static_cast<int>(candidates.size()) - 1;
        for (size_t i = 0; i < candidates.size(); ++i) {
            cum_sum += candidates[i].prob;
            if (cum_sum >= top_p) {
                cutoff_index = static_cast<int>(i);
                break;
            }
        }
        candidates.resize(cutoff_index + 1);

        // 再正規化
        float norm_factor = 0.0f;
        for (const auto& cand : candidates) norm_factor += cand.prob;
        for (auto& cand : candidates) cand.prob /= norm_factor;

        // 累積確率によるサンプリング
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        float r = dis(rng);
        float acc = 0.0f;
        for (const auto& cand : candidates) {
            acc += cand.prob;
            if (r <= acc) {
                return cand.id;
            }
        }

        return candidates.back().id;
    }
};

// ============================================================================
// 6. メイン実行プログラム
// ============================================================================

int main() {
    const int vocab_size = 500;
    const int d_model = 64;
    const int num_layers = 4;
    const int num_heads = 8;
    const int num_kv_heads = 2; // GQA (4 Queries per KV)
    const int intermediate_size = 128;

    std::cout << "=========================================================" << std::endl;
    std::cout << "  Qwen2.5 High-Precision Engine (Pure C++ Implementation)  " << std::endl;
    std::cout << "=========================================================" << std::endl;

    QwenLLM model(vocab_size, d_model, num_layers, num_heads, num_kv_heads, intermediate_size, 2026);

    std::vector<int> prompt = {10, 256, 42, 88, 300};
    std::cout << "\n[Input Prompt Tokens]: ";
    for (int t : prompt) std::cout << t << " ";
    std::cout << std::endl;

    // --- Phase 1: Prefill ---
    std::vector<float> logits;
    model.forward(prompt, logits);

    const float* last_token_logits = logits.data() + (prompt.size() - 1) * vocab_size;
    
    // 高精度 Top-K / Top-P サンプリングの利用
    int next_token = model.sample_advanced(last_token_logits, 0.7f, 40, 0.9f);

    std::cout << "\n--- Prefill Phase Complete ---" << std::endl;
    std::cout << "First Generated Token: " << next_token << std::endl;

    // --- Phase 2: Autoregressive Generation ---
    const int gen_length = 10;
    std::vector<int> generated = prompt;
    generated.push_back(next_token);

    std::cout << "\n--- Generation Phase ---" << std::endl;
    for (int step = 0; step < gen_length; ++step) {
        std::vector<int> step_input = {next_token};
        model.forward(step_input, logits);

        next_token = model.sample_advanced(logits.data(), 0.7f, 40, 0.9f);
        generated.push_back(next_token);

        std::cout << "Step " << std::setw(2) << step + 1 
                  << " | Sampled Token: " << std::setw(4) << next_token 
                  << " | KV Cache Sequence Length: " << model.layer_caches[0].cached_seq_len << std::endl;
    }

    std::cout << "\n[Final Output Sequence]: ";
    for (int id : generated) std::cout << id << " ";
    std::cout << std::endl;

    return 0;
}