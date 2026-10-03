#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <string>

// ============================================================================
// 1. 基礎数学 & 活性化関数 & メモリ演算 (一切の外部ライブラリ不使用)
// ============================================================================

// SiLU (Swish) 活性化関数: x * sigmoid(x)
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// 数値安定化版 Softmax (インプレース計算)
void softmax_inplace(float* x, int size) {
    float max_val = *std::max_element(x, x + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// 1次元展開された行列積演算 Y = X * W
// X: [N x in_f], W: [in_f x out_f], Y: [N x out_f]
void matmul_cpu(const float* X, const float* W, float* Y, int N, int in_f, int out_f) {
    for (int n = 0; n < N; ++n) {
        const float* x_row = X + n * in_f;
        float* y_row = Y + n * out_f;
        for (int j = 0; j < out_f; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < in_f; ++i) {
                sum += x_row[i] * W[i * out_f + j];
            }
            y_row[j] = sum;
        }
    }
}

// ============================================================================
// 2. ニューラルネットワーク基礎レイヤー
// ============================================================================

// 線形層 (Linear Layer - バイアスなし)
class Linear {
public:
    int in_features;
    int out_features;
    std::vector<float> weight; // Shape: [in_features x out_features]

    Linear(int in_f, int out_f, std::mt19937& gen)
        : in_features(in_f), out_features(out_f), weight(in_f * out_f) {
        // Xavier/Glorot 一様分布初期化
        float limit = std::sqrt(6.0f / (in_f + out_f));
        std::uniform_real_distribution<float> dis(-limit, limit);
        for (auto& w : weight) w = dis(gen);
    }

    void forward(const float* input, float* output, int N) const {
        matmul_cpu(input, weight.data(), output, N, in_features, out_features);
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

            float square_sum = 0.0f;
            for (int i = 0; i < dim; ++i) {
                square_sum += in_ptr[i] * in_ptr[i];
            }
            float rms = std::sqrt((square_sum / dim) + eps);
            float inv_rms = 1.0f / rms;

            for (int i = 0; i < dim; ++i) {
                out_ptr[i] = (in_ptr[i] * inv_rms) * weight[i];
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

        // SiLU(gate) * up
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
    // 蓄積型フラットメモリ
    // 形状: [cached_seq_len x num_kv_heads x head_dim]
    std::vector<float> k_cache;
    std::vector<float> v_cache;
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
// 4. Qwen Attention (GQA + RoPE + KV Cache)
// ============================================================================

class QwenAttention {
public:
    int d_model;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int num_queries_per_kv;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;

    QwenAttention(int d_model, int num_heads, int num_kv_heads, std::mt19937& gen)
        : d_model(d_model), num_heads(num_heads), num_kv_heads(num_kv_heads),
          head_dim(d_model / num_heads),
          num_queries_per_kv(num_heads / num_kv_heads),
          q_proj(d_model, num_heads * head_dim, gen),
          k_proj(d_model, num_kv_heads * head_dim, gen),
          v_proj(d_model, num_kv_heads * head_dim, gen),
          out_proj(num_heads * head_dim, d_model, gen) {}

    // 2次元複素回転の位置エンコーディング (RoPE)
    void apply_rope(float* vec, int abs_pos, int dim) const {
        for (int i = 0; i < dim; i += 2) {
            float freq = 1.0f / std::pow(10000.0f, static_cast<float>(i) / dim);
            float theta = abs_pos * freq;
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);

            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i]     = v0 * cos_t - v1 * sin_t;
            vec[i + 1] = v0 * sin_t + v1 * cos_t;
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

        // RoPE を全トークンに適用
        for (int t = 0; t < num_tokens; ++t) {
            int abs_pos = start_pos + t;
            for (int h = 0; h < num_heads; ++h) {
                apply_rope(Q.data() + (t * num_heads + h) * head_dim, abs_pos, head_dim);
            }
            for (int h = 0; h < num_kv_heads; ++h) {
                apply_rope(new_K.data() + (t * num_kv_heads + h) * head_dim, abs_pos, head_dim);
            }
        }

        // KVキャッシュを更新
        kv_cache.append(new_K.data(), new_V.data(), num_tokens, num_kv_heads, head_dim);

        int total_seq_len = kv_cache.cached_seq_len;
        const float* K_full = kv_cache.k_cache.data();
        const float* V_full = kv_cache.v_cache.data();

        std::vector<float> concat_attn_out(num_tokens * num_heads * head_dim, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Grouped-Query Attention (GQA)
        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / num_queries_per_kv;

            for (int i = 0; i < num_tokens; ++i) {
                int current_abs_pos = start_pos + i;
                std::vector<float> attn_scores(total_seq_len, -1e9f);

                // Causal Mask (未来のトークンへのアテンションをマスク)
                for (int j = 0; j <= current_abs_pos; ++j) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        float q_val = Q[(i * num_heads + h) * head_dim + d];
                        float k_val = K_full[(j * num_kv_heads + kv_h) * head_dim + d];
                        score += q_val * k_val;
                    }
                    attn_scores[j] = score * scale;
                }

                // Softmax 正規化
                softmax_inplace(attn_scores.data(), current_abs_pos + 1);

                // Attention Weights * Values
                for (int d = 0; d < head_dim; ++d) {
                    float head_out = 0.0f;
                    for (int j = 0; j <= current_abs_pos; ++j) {
                        float v_val = V_full[(j * num_kv_heads + kv_h) * head_dim + d];
                        head_out += attn_scores[j] * v_val;
                    }
                    concat_attn_out[(i * num_heads + h) * head_dim + d] = head_out;
                }
            }
        }

        out_proj.forward(concat_attn_out.data(), output, num_tokens);
    }
};

// ============================================================================
// 5. Qwen Decoder Block & トランスフォーマーモデル
// ============================================================================

class QwenBlock {
public:
    QwenAttention attn;
    RMSNorm norm1;
    SwiGLUFFN ffn;
    RMSNorm norm2;

    QwenBlock(int d_model, int num_heads, int num_kv_heads, int intermediate_size, std::mt19937& gen)
        : attn(d_model, num_heads, num_kv_heads, gen),
          norm1(d_model),
          ffn(d_model, intermediate_size, gen),
          norm2(d_model) {}

    void forward(const float* input, float* output, int num_tokens, KVCache& kv_cache) const {
        int d_model = attn.d_model;
        int total_size = num_tokens * d_model;

        // 1. Attention (Pre-LN & Residual)
        std::vector<float> norm1_out(total_size);
        norm1.forward(input, norm1_out.data(), num_tokens);

        std::vector<float> attn_out(total_size);
        attn.forward(norm1_out.data(), attn_out.data(), num_tokens, kv_cache);

        std::vector<float> residual1(total_size);
        for (int i = 0; i < total_size; ++i) residual1[i] = input[i] + attn_out[i];

        // 2. FFN (Pre-LN & Residual)
        std::vector<float> norm2_out(total_size);
        norm2.forward(residual1.data(), norm2_out.data(), num_tokens);

        std::vector<float> ffn_out(total_size);
        ffn.forward(norm2_out.data(), ffn_out.data(), num_tokens);

        for (int i = 0; i < total_size; ++i) output[i] = residual1[i] + ffn_out[i];
    }
};

// 全体アーキテクチャ: Embedding -> N x TransformerBlock -> Final RMSNorm -> LM Head
class QwenLLM {
public:
    int vocab_size;
    int d_model;
    int num_layers;
    
    std::vector<float> token_embedding_table; // [vocab_size x d_model]
    std::vector<QwenBlock> layers;
    RMSNorm final_norm;
    Linear lm_head;

    std::vector<KVCache> layer_caches;

    QwenLLM(int vocab_size, int d_model, int num_layers, int num_heads, int num_kv_heads, int intermediate_size, uint32_t seed = 42)
        : vocab_size(vocab_size), d_model(d_model), num_layers(num_layers),
          token_embedding_table(vocab_size * d_model),
          final_norm(d_model),
          lm_head(d_model, vocab_size, std::mt19937(seed + 999)),
          layer_caches(num_layers) {

        std::mt19937 gen(seed);
        float limit = std::sqrt(1.0f / d_model);
        std::uniform_real_distribution<float> dis(-limit, limit);

        for (auto& e : token_embedding_table) e = dis(gen);

        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(d_model, num_heads, num_kv_heads, intermediate_size, gen);
        }
    }

    void reset_cache() {
        for (auto& cache : layer_caches) cache.reset();
    }

    // トークンID配列からLogitsテーブルを算出
    // input_ids: [num_tokens] -> logits: [num_tokens x vocab_size]
    void forward(const std::vector<int>& input_ids, std::vector<float>& logits) {
        int num_tokens = static_cast<int>(input_ids.size());

        // 1. Token Embedding Lookup
        std::vector<float> hidden_states(num_tokens * d_model);
        for (int t = 0; t < num_tokens; ++t) {
            int token_id = input_ids[t];
            const float* emb_ptr = token_embedding_table.data() + token_id * d_model;
            std::copy(emb_ptr, emb_ptr + d_model, hidden_states.begin() + t * d_model);
        }

        // 2. Transformer Layer Stacking
        std::vector<float> layer_output(num_tokens * d_model);
        for (int l = 0; l < num_layers; ++l) {
            layers[l].forward(hidden_states.data(), layer_output.data(), num_tokens, layer_caches[l]);
            hidden_states = layer_output; // 次の層へ受け渡し
        }

        // 3. Final RMSNorm
        std::vector<float> norm_output(num_tokens * d_model);
        final_norm.forward(hidden_states.data(), norm_output.data(), num_tokens);

        // 4. LM Head Projection (Logits 算出)
        logits.resize(num_tokens * vocab_size);
        lm_head.forward(norm_output.data(), logits.data(), num_tokens);
    }

    // Greed / Temperature サンプリングを用いて単語を推論
    int sample_next_token(const float* logits_ptr, float temperature = 0.7f) {
        std::vector<float> temp_logits(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            temp_logits[v] = logits_ptr[v] / temperature;
        }

        softmax_inplace(temp_logits.data(), vocab_size);

        // Greedy (Argmax) 選択の例
        int best_token = 0;
        float max_p = temp_logits[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (temp_logits[v] > max_p) {
                max_p = temp_logits[v];
                best_token = v;
            }
        }
        return best_token;
    }
};

// ============================================================================
// 6. メイン実行プログラム（自己回帰テキスト生成シミュレーション）
// ============================================================================

int main() {
    // LLM ハイパーパラメータの設定
    const int vocab_size = 100;      // 語彙数
    const int d_model = 32;          // 隠れ層の次元数
    const int num_layers = 2;        // トランスフォーマー層数
    const int num_heads = 4;         // Query ヘッド数
    const int num_kv_heads = 2;      // KV ヘッド数 (GQA: 2グループ)
    const int intermediate_size = 64;// FFN 中間層の次元数

    std::cout << "=========================================================" << std::endl;
    std::cout << "  Qwen2.5 Pure C++ Engine (Zero External Dependencies)  " << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Vocab Size:        " << vocab_size << std::endl;
    std::cout << "d_model:           " << d_model << std::endl;
    std::cout << "Layers:            " << num_layers << std::endl;
    std::cout << "Q-Heads / KV-Heads:" << num_heads << " / " << num_kv_heads << std::endl;

    // モデル初期化
    QwenLLM model(vocab_size, d_model, num_layers, num_heads, num_kv_heads, intermediate_size);

    // 入力プロンプト (トークンIDの系列)
    std::vector<int> prompt = {12, 45, 88, 3};
    std::cout << "\n[Input Prompt Tokens]: ";
    for (int t : prompt) std::cout << t << " ";
    std::cout << std::endl;

    // --- Phase 1: Prefill (プロンプトの一括処理 & KVキャッシュ構築) ---
    std::cout << "\n--- Step 1: Prefill Phase ---" << std::endl;
    std::vector<float> logits;
    model.forward(prompt, logits);

    // 最後のトークンのLogitsから最初の生成トークンを得る
    const float* last_token_logits = logits.data() + (prompt.size() - 1) * vocab_size;
    int next_token = model.sample_next_token(last_token_logits);

    std::cout << "Prefill completed. First generated token ID: " << next_token << std::endl;
    std::cout << "Cached Sequence Length in Layer 0: " << model.layer_caches[0].cached_seq_len << std::endl;

    // --- Phase 2: Generation (1トークンずつの自己回帰ループ) ---
    const int max_generate_tokens = 5;
    std::cout << "\n--- Step 2: Autoregressive Generation Loop ---" << std::endl;

    std::vector<int> generated_sequence = prompt;
    generated_sequence.push_back(next_token);

    for (int step = 0; step < max_generate_tokens; ++step) {
        // 次の1トークンのみを渡して計算 (KVキャッシュを利用するため num_tokens = 1)
        std::vector<int> single_token_input = {next_token};
        model.forward(single_token_input, logits);

        // 新しいトークンを選択
        next_token = model.sample_next_token(logits.data());
        generated_sequence.push_back(next_token);

        std::cout << "Gen Step " << step + 1 
                  << " | Sampled Token: " << std::setw(3) << next_token 
                  << " | Total KV Cache Size: " << model.layer_caches[0].cached_seq_len << std::endl;
    }

    // 最終出力結果
    std::cout << "\n[Final Output Sequence]: ";
    for (int id : generated_sequence) {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    return 0;
}