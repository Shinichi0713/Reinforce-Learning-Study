#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <memory>

// ============================================================================
// 1. 基本的な数値演算関数
// ============================================================================

// SiLU (Sigmoid Linear Unit) アクティベーション
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// 数値的に安定した Softmax (インプレース処理)
void safe_softmax_inplace(std::vector<float>& x, size_t size) {
    if (size == 0) return;
    float max_val = *std::max_element(x.begin(), x.begin() + size);
    
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    
    float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    for (size_t i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// 行列積: Y [N x Out] = X [N x In] * W_T [Out x In]^T
// キャッシュ効率向上のため、重み W_T は転置型 [Out x In] で保持
void matmul_cpu(
    const float* x,
    const float* w_t,
    float* y,
    size_t n,
    size_t in_f,
    size_t out_f
) {
    for (size_t i = 0; i < n; ++i) {
        const float* x_row = x + i * in_f;
        float* y_row = y + i * out_f;
        
        for (size_t j = 0; j < out_f; ++j) {
            const float* w_row = w_t + j * in_f;
            double sum = 0.0;
            for (size_t k = 0; k < in_f; ++k) {
                sum += static_cast<double>(x_row[k]) * static_cast<double>(w_row[k]);
            }
            y_row[j] = static_cast<float>(sum);
        }
    }
}

// ============================================================================
// 2. 基本レイヤー実装
// ============================================================================

struct Linear {
    size_t in_features;
    size_t out_features;
    std::vector<float> weight_t; // 転置済み重み: [out_features x in_features]

    Linear(size_t in_f, size_t out_f, std::mt19937& rng)
        : in_features(in_f), out_features(out_f), weight_t(out_f * in_f) {
        // Xavier/He 初期化の近似
        float stddev = std::sqrt(2.0f / static_cast<float>(in_f));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (auto& w : weight_t) {
            w = dist(rng);
        }
    }

    void forward(const float* input, float* output, size_t n) const {
        matmul_cpu(input, weight_t.data(), output, n, in_features, out_features);
    }
};

struct RMSNorm {
    size_t dim;
    float eps;
    std::vector<float> weight;

    RMSNorm(size_t dim, float eps = 1e-6f)
        : dim(dim), eps(eps), weight(dim, 1.0f) {}

    void forward(const float* input, float* output, size_t n) const {
        for (size_t i = 0; i < n; ++i) {
            const float* in_ptr = input + i * dim;
            float* out_ptr = output + i * dim;

            double square_sum = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                square_sum += static_cast<double>(in_ptr[j]) * static_cast<double>(in_ptr[j]);
            }
            
            float rms = std::sqrt(static_cast<float>(square_sum / dim) + eps);
            float inv_rms = 1.0f / rms;

            for (size_t j = 0; j < dim; ++j) {
                out_ptr[j] = in_ptr[j] * inv_rms * weight[j];
            }
        }
    }
};

// ============================================================================
// 3. KV キャッシュ構造体
// ============================================================================

struct KVCache {
    // 形状: [cached_seq_len, num_kv_heads * head_dim]
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    size_t cached_seq_len = 0;

    void reset() {
        k_cache.clear();
        v_cache.clear();
        cached_seq_len = 0;
    }

    // 新規トークンの K, V ベクトルを末尾に追加
    void append(
        const std::vector<float>& new_k,
        const std::vector<float>& new_v,
        size_t num_tokens,
        size_t num_kv_heads,
        size_t head_dim
    ) {
        size_t elements = num_tokens * num_kv_heads * head_dim;
        k_cache.insert(k_cache.end(), new_k.begin(), new_k.begin() + elements);
        v_cache.insert(v_cache.end(), new_v.begin(), new_v.begin() + elements);
        cached_seq_len += num_tokens;
    }
};

// ============================================================================
// 4. KVキャッシュ対応 Multi-Query / Grouped-Query Attention
// ============================================================================

struct Attention {
    size_t d_model;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t num_queries_per_kv;
    float rope_theta;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    Attention(size_t d_model, size_t num_heads, size_t num_kv_heads, std::mt19937& rng, float rope_theta = 10000.0f)
        : d_model(d_model),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_dim(d_model / num_heads),
          num_queries_per_kv(num_heads / num_kv_heads),
          rope_theta(rope_theta),
          q_proj(d_model, num_heads * (d_model / num_heads), rng),
          k_proj(d_model, num_kv_heads * (d_model / num_heads), rng),
          v_proj(d_model, num_kv_heads * (d_model / num_heads), rng),
          out_proj(num_heads * (d_model / num_heads), d_model, rng) {}

    // Rotary Position Embedding (RoPE)
    void apply_rope(float* vec, size_t abs_pos) const {
        for (size_t i = 0; i < head_dim; i += 2) {
            double freq = 1.0 / std::pow(rope_theta, static_cast<double>(i) / head_dim);
            double theta = static_cast<double>(abs_pos) * freq;
            double cos_t = std::cos(theta);
            double sin_t = std::sin(theta);

            float v0 = vec[i];
            float v1 = vec[i + 1];

            vec[i]     = static_cast<float>(v0 * cos_t - v1 * sin_t);
            vec[i + 1] = static_cast<float>(v0 * sin_t + v1 * cos_t);
        }
    }

    void forward(
        const float* input,
        float* output,
        size_t num_tokens,
        KVCache& kv_cache
    ) const {
        size_t start_pos = kv_cache.cached_seq_len;

        std::vector<float> q(num_tokens * num_heads * head_dim);
        std::vector<float> new_k(num_tokens * num_kv_heads * head_dim);
        std::vector<float> new_v(num_tokens * num_kv_heads * head_dim);

        // 1. Q, K, V プロジェクション
        q_proj.forward(input, q.data(), num_tokens);
        k_proj.forward(input, new_k.data(), num_tokens);
        v_proj.forward(input, new_v.data(), num_tokens);

        // 2. 各 Head への RoPE 適用
        for (size_t t = 0; t < num_tokens; ++t) {
            size_t abs_pos = start_pos + t;
            for (size_t h = 0; h < num_heads; ++h) {
                size_t offset = (t * num_heads + h) * head_dim;
                apply_rope(q.data() + offset, abs_pos);
            }
            for (size_t h = 0; h < num_kv_heads; ++h) {
                size_t offset = (t * num_kv_heads + h) * head_dim;
                apply_rope(new_k.data() + offset, abs_pos);
            }
        }

        // 3. 新しい Key/Value を Cache へ蓄積 (KV Cache 更新)
        kv_cache.append(new_k, new_v, num_tokens, num_kv_heads, head_dim);

        size_t total_seq_len = kv_cache.cached_seq_len;
        const float* k_full = kv_cache.k_cache.data();
        const float* v_full = kv_cache.v_cache.data();

        std::vector<float> concat_attn_out(num_tokens * num_heads * head_dim, 0.0f);
        std::vector<float> attn_scores(total_seq_len, -1e9f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // 4. Scaled Dot-Product Attention (KV Cache 全体を参照)
        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / num_queries_per_kv; // GQA インデックス計算

            for (size_t i = 0; i < num_tokens; ++i) {
                size_t current_abs_pos = start_pos + i;

                // 過去のすべてのトークン (0 .. current_abs_pos) に対してスコア計算
                for (size_t j = 0; j <= current_abs_pos; ++j) {
                    size_t q_offset = (i * num_heads + h) * head_dim;
                    size_t k_offset = (j * num_kv_heads + kv_h) * head_dim;

                    const float* q_ptr = q.data() + q_offset;
                    const float* k_ptr = k_full + k_offset;

                    double score = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        score += static_cast<double>(q_ptr[d]) * static_cast<double>(k_ptr[d]);
                    }
                    attn_scores[j] = static_cast<float>(score) * scale;
                }

                // Causal Softmax (未来のトークンへのアテンションをマスク)
                safe_softmax_inplace(attn_scores, current_abs_pos + 1);

                // Value との重み付き和
                for (size_t d = 0; d < head_dim; ++d) {
                    double head_out = 0.0;
                    for (size_t j = 0; j <= current_abs_pos; ++j) {
                        size_t v_offset = (j * num_kv_heads + kv_h) * head_dim;
                        head_out += static_cast<double>(attn_scores[j]) * static_cast<double>(v_full[v_offset + d]);
                    }
                    size_t out_offset = (i * num_heads + h) * head_dim + d;
                    concat_attn_out[out_offset] = static_cast<float>(head_out);
                }
            }
        }

        // 5. Output プロジェクション
        out_proj.forward(concat_attn_out.data(), output, num_tokens);
    }
};

// ============================================================================
// 5. Transformer ブロック & 全体言語モデル構造
// ============================================================================

struct SwiGLUFFN {
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;

    SwiGLUFFN(size_t d_model, size_t hidden_dim, std::mt19937& rng)
        : gate_proj(d_model, hidden_dim, rng),
          up_proj(d_model, hidden_dim, rng),
          down_proj(hidden_dim, d_model, rng) {}

    void forward(const float* input, float* output, size_t num_tokens) const {
        size_t hidden_dim = gate_proj.out_features;
        std::vector<float> gate_out(num_tokens * hidden_dim);
        std::vector<float> up_out(num_tokens * hidden_dim);
        std::vector<float> act_out(num_tokens * hidden_dim);

        gate_proj.forward(input, gate_out.data(), num_tokens);
        up_proj.forward(input, up_out.data(), num_tokens);

        for (size_t i = 0; i < num_tokens * hidden_dim; ++i) {
            act_out[i] = silu(gate_out[i]) * up_out[i];
        }

        down_proj.forward(act_out.data(), output, num_tokens);
    }
};

struct TransformerBlock {
    RMSNorm norm1;
    Attention attn;
    RMSNorm norm2;
    SwiGLUFFN ffn;

    TransformerBlock(size_t d_model, size_t num_heads, size_t num_kv_heads, size_t intermediate_size, std::mt19937& rng)
        : norm1(d_model),
          attn(d_model, num_heads, num_kv_heads, rng),
          norm2(d_model),
          ffn(d_model, intermediate_size, rng) {}

    void forward(
        const float* input,
        float* output,
        size_t num_tokens,
        KVCache& kv_cache
    ) const {
        size_t d_model = norm1.dim;
        size_t total_size = num_tokens * d_model;

        std::vector<float> norm1_out(total_size);
        std::vector<float> attn_out(total_size);
        std::vector<float> residual1(total_size);

        norm1.forward(input, norm1_out.data(), num_tokens);
        attn.forward(norm1_out.data(), attn_out.data(), num_tokens, kv_cache);

        for (size_t i = 0; i < total_size; ++i) {
            residual1[i] = input[i] + attn_out[i];
        }

        std::vector<float> norm2_out(total_size);
        std::vector<float> ffn_out(total_size);

        norm2.forward(residual1.data(), norm2_out.data(), num_tokens);
        ffn.forward(norm2_out.data(), ffn_out.data(), num_tokens);

        for (size_t i = 0; i < total_size; ++i) {
            output[i] = residual1[i] + ffn_out[i];
        }
    }
};

class SimpleLLM {
public:
    size_t vocab_size;
    size_t d_model;
    size_t num_layers;

    std::vector<float> token_embedding_table;
    std::vector<TransformerBlock> layers;
    std::vector<KVCache> layer_caches;
    RMSNorm final_norm;
    Linear lm_head;

    SimpleLLM(
        size_t vocab_size,
        size_t d_model,
        size_t num_layers,
        size_t num_heads,
        size_t num_kv_heads,
        size_t intermediate_size,
        unsigned int seed = 42
    ) : vocab_size(vocab_size),
        d_model(d_model),
        num_layers(num_layers),
        token_embedding_table(vocab_size * d_model),
        final_norm(d_model),
        lm_head(d_model, vocab_size, rng_),
        rng_(seed) {

        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(d_model)));
        for (auto& e : token_embedding_table) {
            e = dist(rng_);
        }

        for (size_t i = 0; i < num_layers; ++i) {
            layers.emplace_back(d_model, num_heads, num_kv_heads, intermediate_size, rng_);
            layer_caches.emplace_back();
        }
    }

    void reset_cache() {
        for (auto& cache : layer_caches) {
            cache.reset();
        }
    }

    void forward(const std::vector<size_t>& input_ids, std::vector<float>& logits) {
        size_t num_tokens = input_ids.size();
        size_t total_emb_size = num_tokens * d_model;

        std::vector<float> hidden_states(total_emb_size);

        // Embedding ルックアップ
        for (size_t t = 0; t < num_tokens; ++t) {
            size_t token_id = input_ids[t];
            const float* emb_ptr = token_embedding_table.data() + token_id * d_model;
            std::copy(emb_ptr, emb_ptr + d_model, hidden_states.data() + t * d_model);
        }

        std::vector<float> layer_output(total_emb_size);

        // 各 TransformerBlock の順伝播
        for (size_t l = 0; l < num_layers; ++l) {
            layers[l].forward(hidden_states.data(), layer_output.data(), num_tokens, layer_caches[l]);
            hidden_states = layer_output;
        }

        std::vector<float> norm_output(total_emb_size);
        final_norm.forward(hidden_states.data(), norm_output.data(), num_tokens);

        logits.resize(num_tokens * vocab_size);
        lm_head.forward(norm_output.data(), logits.data(), num_tokens);
    }

    // Argmax サンプリング
    size_t sample_greedy(const float* logits_ptr) {
        return std::distance(logits_ptr, std::max_element(logits_ptr, logits_ptr + vocab_size));
    }

private:
    std::mt19937 rng_;
};

// ============================================================================
// 6. メイン実行プログラム (Prefill + Autoregressive 世代ループ)
// ============================================================================

int main() {
    size_t vocab_size = 500;
    size_t d_model = 64;
    size_t num_layers = 4;
    size_t num_heads = 8;
    size_t num_kv_heads = 2; // GQA
    size_t intermediate_size = 128;

    std::cout << "=========================================================\n";
    std::cout << "  C++ LLM Inference Engine with KV-Cache\n";
    std::cout << "=========================================================\n\n";

    SimpleLLM model(vocab_size, d_model, num_layers, num_heads, num_kv_heads, intermediate_size, 2026);

    std::vector<size_t> prompt = {10, 256, 42, 88, 300};
    std::cout << "[Input Prompt Tokens]: ";
    for (size_t id : prompt) std::cout << id << " ";
    std::cout << "\n\n";

    // --- Phase 1: Prefill (プロンプトを一括処理) ---
    std::vector<float> logits;
    model.forward(prompt, logits);

    // 最後のトークンの出力 Logits から次のトークンをサンプリング
    const float* last_token_logits = logits.data() + (prompt.size() - 1) * vocab_size;
    size_t next_token = model.sample_greedy(last_token_logits);

    std::cout << "--- Prefill Phase Complete ---\n";
    std::cout << "First Generated Token: " << next_token << "\n\n";

    // --- Phase 2: Autoregressive Generation (KVキャッシュを利用して1トークンずつ生成) ---
    size_t gen_length = 10;
    std::vector<size_t> generated = prompt;
    generated.push_back(next_token);

    std::cout << "--- Generation Phase (1 Token per Step) ---\n";
    for (size_t step = 0; step < gen_length; ++step) {
        // KVキャッシュがあるため、入力は「直前の1トークンのみ」
        std::vector<size_t> step_input = {next_token};
        model.forward(step_input, logits);

        next_token = model.sample_greedy(logits.data());
        generated.push_back(next_token);

        std::cout << "Step " << (step + 1) 
                  << " | Sampled Token: " << next_token 
                  << " | KV Cache Length: " << model.layer_caches[0].cached_seq_len << "\n";
    }

    std::cout << "\n[Final Generated Sequence]: ";
    for (size_t id : generated) std::cout << id << " ";
    std::cout << "\n";

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <memory>

// ============================================================================
// 1. 基本的な数値演算関数
// ============================================================================

// SiLU (Sigmoid Linear Unit) アクティベーション
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// 数値的に安定した Softmax (インプレース処理)
void safe_softmax_inplace(std::vector<float>& x, size_t size) {
    if (size == 0) return;
    float max_val = *std::max_element(x.begin(), x.begin() + size);
    
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    
    float inv_sum = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
    for (size_t i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// 行列積: Y [N x Out] = X [N x In] * W_T [Out x In]^T
// キャッシュ効率向上のため、重み W_T は転置型 [Out x In] で保持
void matmul_cpu(
    const float* x,
    const float* w_t,
    float* y,
    size_t n,
    size_t in_f,
    size_t out_f
) {
    for (size_t i = 0; i < n; ++i) {
        const float* x_row = x + i * in_f;
        float* y_row = y + i * out_f;
        
        for (size_t j = 0; j < out_f; ++j) {
            const float* w_row = w_t + j * in_f;
            double sum = 0.0;
            for (size_t k = 0; k < in_f; ++k) {
                sum += static_cast<double>(x_row[k]) * static_cast<double>(w_row[k]);
            }
            y_row[j] = static_cast<float>(sum);
        }
    }
}

// ============================================================================
// 2. 基本レイヤー実装
// ============================================================================

struct Linear {
    size_t in_features;
    size_t out_features;
    std::vector<float> weight_t; // 転置済み重み: [out_features x in_features]

    Linear(size_t in_f, size_t out_f, std::mt19937& rng)
        : in_features(in_f), out_features(out_f), weight_t(out_f * in_f) {
        // Xavier/He 初期化の近似
        float stddev = std::sqrt(2.0f / static_cast<float>(in_f));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (auto& w : weight_t) {
            w = dist(rng);
        }
    }

    void forward(const float* input, float* output, size_t n) const {
        matmul_cpu(input, weight_t.data(), output, n, in_features, out_features);
    }
};

struct RMSNorm {
    size_t dim;
    float eps;
    std::vector<float> weight;

    RMSNorm(size_t dim, float eps = 1e-6f)
        : dim(dim), eps(eps), weight(dim, 1.0f) {}

    void forward(const float* input, float* output, size_t n) const {
        for (size_t i = 0; i < n; ++i) {
            const float* in_ptr = input + i * dim;
            float* out_ptr = output + i * dim;

            double square_sum = 0.0;
            for (size_t j = 0; j < dim; ++j) {
                square_sum += static_cast<double>(in_ptr[j]) * static_cast<double>(in_ptr[j]);
            }
            
            float rms = std::sqrt(static_cast<float>(square_sum / dim) + eps);
            float inv_rms = 1.0f / rms;

            for (size_t j = 0; j < dim; ++j) {
                out_ptr[j] = in_ptr[j] * inv_rms * weight[j];
            }
        }
    }
};

// ============================================================================
// 3. KV キャッシュ構造体
// ============================================================================

struct KVCache {
    // 形状: [cached_seq_len, num_kv_heads * head_dim]
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    size_t cached_seq_len = 0;

    void reset() {
        k_cache.clear();
        v_cache.clear();
        cached_seq_len = 0;
    }

    // 新規トークンの K, V ベクトルを末尾に追加
    void append(
        const std::vector<float>& new_k,
        const std::vector<float>& new_v,
        size_t num_tokens,
        size_t num_kv_heads,
        size_t head_dim
    ) {
        size_t elements = num_tokens * num_kv_heads * head_dim;
        k_cache.insert(k_cache.end(), new_k.begin(), new_k.begin() + elements);
        v_cache.insert(v_cache.end(), new_v.begin(), new_v.begin() + elements);
        cached_seq_len += num_tokens;
    }
};

// ============================================================================
// 4. KVキャッシュ対応 Multi-Query / Grouped-Query Attention
// ============================================================================

struct Attention {
    size_t d_model;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_dim;
    size_t num_queries_per_kv;
    float rope_theta;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    Attention(size_t d_model, size_t num_heads, size_t num_kv_heads, std::mt19937& rng, float rope_theta = 10000.0f)
        : d_model(d_model),
          num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          head_dim(d_model / num_heads),
          num_queries_per_kv(num_heads / num_kv_heads),
          rope_theta(rope_theta),
          q_proj(d_model, num_heads * (d_model / num_heads), rng),
          k_proj(d_model, num_kv_heads * (d_model / num_heads), rng),
          v_proj(d_model, num_kv_heads * (d_model / num_heads), rng),
          out_proj(num_heads * (d_model / num_heads), d_model, rng) {}

    // Rotary Position Embedding (RoPE)
    void apply_rope(float* vec, size_t abs_pos) const {
        for (size_t i = 0; i < head_dim; i += 2) {
            double freq = 1.0 / std::pow(rope_theta, static_cast<double>(i) / head_dim);
            double theta = static_cast<double>(abs_pos) * freq;
            double cos_t = std::cos(theta);
            double sin_t = std::sin(theta);

            float v0 = vec[i];
            float v1 = vec[i + 1];

            vec[i]     = static_cast<float>(v0 * cos_t - v1 * sin_t);
            vec[i + 1] = static_cast<float>(v0 * sin_t + v1 * cos_t);
        }
    }

    void forward(
        const float* input,
        float* output,
        size_t num_tokens,
        KVCache& kv_cache
    ) const {
        size_t start_pos = kv_cache.cached_seq_len;

        std::vector<float> q(num_tokens * num_heads * head_dim);
        std::vector<float> new_k(num_tokens * num_kv_heads * head_dim);
        std::vector<float> new_v(num_tokens * num_kv_heads * head_dim);

        // 1. Q, K, V プロジェクション
        q_proj.forward(input, q.data(), num_tokens);
        k_proj.forward(input, new_k.data(), num_tokens);
        v_proj.forward(input, new_v.data(), num_tokens);

        // 2. 各 Head への RoPE 適用
        for (size_t t = 0; t < num_tokens; ++t) {
            size_t abs_pos = start_pos + t;
            for (size_t h = 0; h < num_heads; ++h) {
                size_t offset = (t * num_heads + h) * head_dim;
                apply_rope(q.data() + offset, abs_pos);
            }
            for (size_t h = 0; h < num_kv_heads; ++h) {
                size_t offset = (t * num_kv_heads + h) * head_dim;
                apply_rope(new_k.data() + offset, abs_pos);
            }
        }

        // 3. 新しい Key/Value を Cache へ蓄積 (KV Cache 更新)
        kv_cache.append(new_k, new_v, num_tokens, num_kv_heads, head_dim);

        size_t total_seq_len = kv_cache.cached_seq_len;
        const float* k_full = kv_cache.k_cache.data();
        const float* v_full = kv_cache.v_cache.data();

        std::vector<float> concat_attn_out(num_tokens * num_heads * head_dim, 0.0f);
        std::vector<float> attn_scores(total_seq_len, -1e9f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // 4. Scaled Dot-Product Attention (KV Cache 全体を参照)
        for (size_t h = 0; h < num_heads; ++h) {
            size_t kv_h = h / num_queries_per_kv; // GQA インデックス計算

            for (size_t i = 0; i < num_tokens; ++i) {
                size_t current_abs_pos = start_pos + i;

                // 過去のすべてのトークン (0 .. current_abs_pos) に対してスコア計算
                for (size_t j = 0; j <= current_abs_pos; ++j) {
                    size_t q_offset = (i * num_heads + h) * head_dim;
                    size_t k_offset = (j * num_kv_heads + kv_h) * head_dim;

                    const float* q_ptr = q.data() + q_offset;
                    const float* k_ptr = k_full + k_offset;

                    double score = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        score += static_cast<double>(q_ptr[d]) * static_cast<double>(k_ptr[d]);
                    }
                    attn_scores[j] = static_cast<float>(score) * scale;
                }

                // Causal Softmax (未来のトークンへのアテンションをマスク)
                safe_softmax_inplace(attn_scores, current_abs_pos + 1);

                // Value との重み付き和
                for (size_t d = 0; d < head_dim; ++d) {
                    double head_out = 0.0;
                    for (size_t j = 0; j <= current_abs_pos; ++j) {
                        size_t v_offset = (j * num_kv_heads + kv_h) * head_dim;
                        head_out += static_cast<double>(attn_scores[j]) * static_cast<double>(v_full[v_offset + d]);
                    }
                    size_t out_offset = (i * num_heads + h) * head_dim + d;
                    concat_attn_out[out_offset] = static_cast<float>(head_out);
                }
            }
        }

        // 5. Output プロジェクション
        out_proj.forward(concat_attn_out.data(), output, num_tokens);
    }
};

// ============================================================================
// 5. Transformer ブロック & 全体言語モデル構造
// ============================================================================

struct SwiGLUFFN {
    Linear gate_proj;
    Linear up_proj;
    Linear down_proj;

    SwiGLUFFN(size_t d_model, size_t hidden_dim, std::mt19937& rng)
        : gate_proj(d_model, hidden_dim, rng),
          up_proj(d_model, hidden_dim, rng),
          down_proj(hidden_dim, d_model, rng) {}

    void forward(const float* input, float* output, size_t num_tokens) const {
        size_t hidden_dim = gate_proj.out_features;
        std::vector<float> gate_out(num_tokens * hidden_dim);
        std::vector<float> up_out(num_tokens * hidden_dim);
        std::vector<float> act_out(num_tokens * hidden_dim);

        gate_proj.forward(input, gate_out.data(), num_tokens);
        up_proj.forward(input, up_out.data(), num_tokens);

        for (size_t i = 0; i < num_tokens * hidden_dim; ++i) {
            act_out[i] = silu(gate_out[i]) * up_out[i];
        }

        down_proj.forward(act_out.data(), output, num_tokens);
    }
};

struct TransformerBlock {
    RMSNorm norm1;
    Attention attn;
    RMSNorm norm2;
    SwiGLUFFN ffn;

    TransformerBlock(size_t d_model, size_t num_heads, size_t num_kv_heads, size_t intermediate_size, std::mt19937& rng)
        : norm1(d_model),
          attn(d_model, num_heads, num_kv_heads, rng),
          norm2(d_model),
          ffn(d_model, intermediate_size, rng) {}

    void forward(
        const float* input,
        float* output,
        size_t num_tokens,
        KVCache& kv_cache
    ) const {
        size_t d_model = norm1.dim;
        size_t total_size = num_tokens * d_model;

        std::vector<float> norm1_out(total_size);
        std::vector<float> attn_out(total_size);
        std::vector<float> residual1(total_size);

        norm1.forward(input, norm1_out.data(), num_tokens);
        attn.forward(norm1_out.data(), attn_out.data(), num_tokens, kv_cache);

        for (size_t i = 0; i < total_size; ++i) {
            residual1[i] = input[i] + attn_out[i];
        }

        std::vector<float> norm2_out(total_size);
        std::vector<float> ffn_out(total_size);

        norm2.forward(residual1.data(), norm2_out.data(), num_tokens);
        ffn.forward(norm2_out.data(), ffn_out.data(), num_tokens);

        for (size_t i = 0; i < total_size; ++i) {
            output[i] = residual1[i] + ffn_out[i];
        }
    }
};

class SimpleLLM {
public:
    size_t vocab_size;
    size_t d_model;
    size_t num_layers;

    std::vector<float> token_embedding_table;
    std::vector<TransformerBlock> layers;
    std::vector<KVCache> layer_caches;
    RMSNorm final_norm;
    Linear lm_head;

    SimpleLLM(
        size_t vocab_size,
        size_t d_model,
        size_t num_layers,
        size_t num_heads,
        size_t num_kv_heads,
        size_t intermediate_size,
        unsigned int seed = 42
    ) : vocab_size(vocab_size),
        d_model(d_model),
        num_layers(num_layers),
        token_embedding_table(vocab_size * d_model),
        final_norm(d_model),
        lm_head(d_model, vocab_size, rng_),
        rng_(seed) {

        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(d_model)));
        for (auto& e : token_embedding_table) {
            e = dist(rng_);
        }

        for (size_t i = 0; i < num_layers; ++i) {
            layers.emplace_back(d_model, num_heads, num_kv_heads, intermediate_size, rng_);
            layer_caches.emplace_back();
        }
    }

    void reset_cache() {
        for (auto& cache : layer_caches) {
            cache.reset();
        }
    }

    void forward(const std::vector<size_t>& input_ids, std::vector<float>& logits) {
        size_t num_tokens = input_ids.size();
        size_t total_emb_size = num_tokens * d_model;

        std::vector<float> hidden_states(total_emb_size);

        // Embedding ルックアップ
        for (size_t t = 0; t < num_tokens; ++t) {
            size_t token_id = input_ids[t];
            const float* emb_ptr = token_embedding_table.data() + token_id * d_model;
            std::copy(emb_ptr, emb_ptr + d_model, hidden_states.data() + t * d_model);
        }

        std::vector<float> layer_output(total_emb_size);

        // 各 TransformerBlock の順伝播
        for (size_t l = 0; l < num_layers; ++l) {
            layers[l].forward(hidden_states.data(), layer_output.data(), num_tokens, layer_caches[l]);
            hidden_states = layer_output;
        }

        std::vector<float> norm_output(total_emb_size);
        final_norm.forward(hidden_states.data(), norm_output.data(), num_tokens);

        logits.resize(num_tokens * vocab_size);
        lm_head.forward(norm_output.data(), logits.data(), num_tokens);
    }

    // Argmax サンプリング
    size_t sample_greedy(const float* logits_ptr) {
        return std::distance(logits_ptr, std::max_element(logits_ptr, logits_ptr + vocab_size));
    }

private:
    std::mt19937 rng_;
};

// ============================================================================
// 6. メイン実行プログラム (Prefill + Autoregressive 世代ループ)
// ============================================================================

int main() {
    size_t vocab_size = 500;
    size_t d_model = 64;
    size_t num_layers = 4;
    size_t num_heads = 8;
    size_t num_kv_heads = 2; // GQA
    size_t intermediate_size = 128;

    std::cout << "=========================================================\n";
    std::cout << "  C++ LLM Inference Engine with KV-Cache\n";
    std::cout << "=========================================================\n\n";

    SimpleLLM model(vocab_size, d_model, num_layers, num_heads, num_kv_heads, intermediate_size, 2026);

    std::vector<size_t> prompt = {10, 256, 42, 88, 300};
    std::cout << "[Input Prompt Tokens]: ";
    for (size_t id : prompt) std::cout << id << " ";
    std::cout << "\n\n";

    // --- Phase 1: Prefill (プロンプトを一括処理) ---
    std::vector<float> logits;
    model.forward(prompt, logits);

    // 最後のトークンの出力 Logits から次のトークンをサンプリング
    const float* last_token_logits = logits.data() + (prompt.size() - 1) * vocab_size;
    size_t next_token = model.sample_greedy(last_token_logits);

    std::cout << "--- Prefill Phase Complete ---\n";
    std::cout << "First Generated Token: " << next_token << "\n\n";

    // --- Phase 2: Autoregressive Generation (KVキャッシュを利用して1トークンずつ生成) ---
    size_t gen_length = 10;
    std::vector<size_t> generated = prompt;
    generated.push_back(next_token);

    std::cout << "--- Generation Phase (1 Token per Step) ---\n";
    for (size_t step = 0; step < gen_length; ++step) {
        // KVキャッシュがあるため、入力は「直前の1トークンのみ」
        std::vector<size_t> step_input = {next_token};
        model.forward(step_input, logits);

        next_token = model.sample_greedy(logits.data());
        generated.push_back(next_token);

        std::cout << "Step " << (step + 1) 
                  << " | Sampled Token: " << next_token 
                  << " | KV Cache Length: " << model.layer_caches[0].cached_seq_len << "\n";
    }

    std::cout << "\n[Final Generated Sequence]: ";
    for (size_t id : generated) std::cout << id << " ";
    std::cout << "\n";

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

class RotaryPositionEmbedding {
private:
    size_t dim;            // 各ヘッドの次元数 (d_model / num_heads)
    size_t max_seq_len;    // 事前計算する最大シーケンス長
    float base;            // 周波数計算のベース値 (通常 10000.0)

    // 回転行列用の Cos / Sin キャッシュ
    // 形状: [max_seq_len, dim / 2]
    std::vector<float> cos_cache;
    std::vector<float> sin_cache;

    void precompute_freqs() {
        size_t half_dim = dim / 2;
        cos_cache.resize(max_seq_len * half_dim);
        sin_cache.resize(max_seq_len * half_dim);

        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (size_t i = 0; i < half_dim; ++i) {
                // theta_i = base ^ (-2i / dim)
                float freq = 1.0f / std::pow(base, static_cast<float>(2 * i) / static_cast<float>(dim));
                float val = static_cast<float>(pos) * freq;

                size_t idx = pos * half_dim + i;
                cos_cache[idx] = std::cos(val);
                sin_cache[idx] = std::sin(val);
            }
        }
    }

public:
    RotaryPositionEmbedding(size_t dim, size_t max_seq_len = 2048, float base = 10000.0f)
        : dim(dim), max_seq_len(max_seq_len), base(base) {
        if (dim % 2 != 0) {
            throw std::invalid_argument("Dimension must be even for RoPE.");
        }
        precompute_freqs();
    }

    // クエリ/キーテンソルに RoPE を適用するメソッド
    // x: [seq_len, num_heads, dim] をフラットにした1次元配列
    void forward(std::vector<float>& x, size_t seq_len, size_t num_heads, size_t start_pos = 0) const {
        if (start_pos + seq_len > max_seq_len) {
            throw std::out_of_range("Sequence length exceeds precomputed max_seq_len.");
        }

        size_t half_dim = dim / 2;

        for (size_t pos = 0; pos < seq_len; ++pos) {
            size_t cache_pos = start_pos + pos;
            size_t cache_offset = cache_pos * half_dim;

            for (size_t h = 0; h < num_heads; ++h) {
                size_t base_idx = (pos * num_heads + h) * dim;

                for (size_t i = 0; i < half_dim; ++i) {
                    float cos_val = cos_cache[cache_offset + i];
                    float sin_val = sin_cache[cache_offset + i];

                    // ペアとなる成分 x[2i] と x[2i+1] を回転させる
                    float x0 = x[base_idx + 2 * i];
                    float x1 = x[base_idx + 2 * i + 1];

                    x[base_idx + 2 * i]     = x0 * cos_val - x1 * sin_val;
                    x[base_idx + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
};

int main() {
    constexpr size_t seq_len = 2;
    constexpr size_t num_heads = 2;
    constexpr size_t head_dim = 4;

    // ダミー入力データ [seq_len=2, num_heads=2, head_dim=4]
    std::vector<float> query = {
        // Position 0
        1.0f, 0.0f, 2.0f, 1.0f,  // Head 0
        0.5f, 1.5f, 0.0f, 1.0f,  // Head 1
        // Position 1
        1.0f, 0.0f, 2.0f, 1.0f,  // Head 0
        0.5f, 1.5f, 0.0f, 1.0f   // Head 1
    };

    RotaryPositionEmbedding rope(head_dim, 512);

    std::cout << "--- Before RoPE ---" << std::endl;
    std::cout << "Pos 1, Head 0: [" << query[8] << ", " << query[9] << ", " << query[10] << ", " << query[11] << "]" << std::endl;

    // RoPEの適用（インプレース変換）
    rope.forward(query, seq_len, num_heads);

    std::cout << "\n--- After RoPE ---" << std::endl;
    std::cout << "Pos 0, Head 0: [" << query[0] << ", " << query[1] << ", " << query[2] << ", " << query[3] << "]" << std::endl;
    std::cout << "Pos 1, Head 0: [" << query[8] << ", " << query[9] << ", " << query[10] << ", " << query[11] << "]" << std::endl;

    return 0;
}