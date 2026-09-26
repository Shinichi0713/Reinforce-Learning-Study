#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// ----------------------------------------------------------------------
// 1. Qwen2 RMSNorm
// ----------------------------------------------------------------------
struct Qwen2RMSNormImpl : torch::nn::Module {
    double eps;
    torch::Tensor weight;

    Qwen2RMSNormImpl(int64_t hidden_size, double eps = 1e-6) : eps(eps) {
        weight = register_parameter("weight", torch::ones({hidden_size}));
    }

    torch::Tensor forward(torch::Tensor hidden_states) {
        auto input_dtype = hidden_states.scalar_type();
        auto states_f32 = hidden_states.to(torch::kFloat32);
        auto variance = states_f32.pow(2).mean(-1, /*keepdim=*/true);
        auto norm_states = states_f32 * torch::rsqrt(variance + eps);
        return (weight * norm_states).to(input_dtype);
    }
};
TORCH_MODULE(Qwen2RMSNorm);

// ----------------------------------------------------------------------
// 2. Rotary Position Embedding (RoPE)
// ----------------------------------------------------------------------
struct Qwen2RotaryEmbeddingImpl : torch::nn::Module {
    int64_t dim;
    double base;
    torch::Tensor inv_freq;

    Qwen2RotaryEmbeddingImpl(int64_t dim, double base = 10000.0)
        : dim(dim), base(base) {
        auto torch_arange = torch::arange(0, dim, 2, torch::kFloat32);
        inv_freq = register_buffer("inv_freq", 1.0 / torch::pow(base, torch_arange / static_cast<double>(dim)));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(int64_t seq_len, torch::Device device) {
        auto t = torch::arange(seq_len, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto freqs = torch::outer(t, inv_freq);
        auto emb = torch::cat({freqs, freqs}, -1);
        return {emb.cos(), emb.sin()};
    }
};
TORCH_MODULE(Qwen2RotaryEmbedding);

// RoPE ユーティリティ関数
inline torch::Tensor rotate_half(const torch::Tensor& x) {
    int64_t half_dim = x.size(-1) / 2;
    auto x1 = x.slice(-1, 0, half_dim);
    auto x2 = x.slice(-1, half_dim, x.size(-1));
    return torch::cat({-x2, x1}, -1);
}

inline std::pair<torch::Tensor, torch::Tensor> apply_rotary_pos_emb(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& cos, const torch::Tensor& sin) {
    
    // (seq_len, dim) -> (1, 1, seq_len, dim)
    auto cos_exp = cos.unsqueeze(0).unsqueeze(1);
    auto sin_exp = sin.unsqueeze(0).unsqueeze(1);

    auto q_embed = (q * cos_exp) + (rotate_half(q) * sin_exp);
    auto k_embed = (k * cos_exp) + (rotate_half(k) * sin_exp);
    return {q_embed, k_embed};
}

// ----------------------------------------------------------------------
// 3. Grouped-Query Attention (GQA) with QKV Bias
// ----------------------------------------------------------------------
struct Qwen2AttentionImpl : torch::nn::Module {
    int64_t hidden_size;
    int64_t num_heads;
    int64_t num_key_value_heads;
    int64_t head_dim;
    int64_t num_key_value_groups;

    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, o_proj{nullptr};

    Qwen2AttentionImpl(int64_t hidden_size, int64_t num_heads, int64_t num_key_value_heads)
        : hidden_size(hidden_size), num_heads(num_heads), num_key_value_heads(num_key_value_heads),
          head_dim(hidden_size / num_heads), num_key_value_groups(num_heads / num_key_value_heads) {

        // Qwen 特有の QKV Bias=True 設定
        q_proj = register_module("q_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, num_heads * head_dim).bias(true)));
        k_proj = register_module("k_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, num_key_value_heads * head_dim).bias(true)));
        v_proj = register_module("v_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, num_key_value_heads * head_dim).bias(true)));
        o_proj = register_module("o_proj", torch::nn::Linear(torch::nn::LinearOptions(num_heads * head_dim, hidden_size).bias(false)));
    }

    torch::Tensor forward(torch::Tensor hidden_states, Qwen2RotaryEmbedding& rotary_emb, torch::Tensor attention_mask = {}) {
        int64_t bsz = hidden_states.size(0);
        int64_t q_len = hidden_states.size(1);

        auto q = q_proj(hidden_states).view({bsz, q_len, num_heads, head_dim}).transpose(1, 2);
        auto k = k_proj(hidden_states).view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);
        auto v = v_proj(hidden_states).view({bsz, q_len, num_key_value_heads, head_dim}).transpose(1, 2);

        auto [cos, sin] = rotary_emb->forward(q_len, hidden_states.device());
        auto [q_emb, k_emb] = apply_rotary_pos_emb(q, k, cos, sin);

        // GQA: Key/Value ヘッドを Query ヘッド数に合わせてリピート拡張
        if (num_key_value_groups > 1) {
            k_emb = k_emb.repeat_interleave(num_key_value_groups, 1);
            v = v.repeat_interleave(num_key_value_groups, 1);
        }

        // Scaled Dot-Product Attention 手動演算 (Causal Mask 対応)
        double scale = 1.0 / std::sqrt(head_dim);
        auto scores = torch::matmul(q_emb, k_emb.transpose(-2, -1)) * scale;

        // Causal Mask の適用
        auto causal_mask = torch::full({q_len, q_len}, -1e9, scores.options()).triu(1);
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0);

        if (attention_mask.defined()) {
            scores = scores + attention_mask;
        }

        auto attn_weights = torch::softmax(scores, -1);
        auto attn_output = torch::matmul(attn_weights, v);

        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, q_len, hidden_size});
        return o_proj(attn_output);
    }
};
TORCH_MODULE(Qwen2Attention);

// ----------------------------------------------------------------------
// 4. SwiGLU MLP
// ----------------------------------------------------------------------
struct Qwen2MLPImpl : torch::nn::Module {
    torch::nn::Linear gate_proj{nullptr}, up_proj{nullptr}, down_proj{nullptr};

    Qwen2MLPImpl(int64_t hidden_size, int64_t intermediate_size) {
        gate_proj = register_module("gate_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)));
        up_proj = register_module("up_proj", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, intermediate_size).bias(false)));
        down_proj = register_module("down_proj", torch::nn::Linear(torch::nn::LinearOptions(intermediate_size, hidden_size).bias(false)));
    }

    torch::Tensor forward(torch::Tensor x) {
        return down_proj(torch::silu(gate_proj(x)) * up_proj(x));
    }
};
TORCH_MODULE(Qwen2MLP);

// ----------------------------------------------------------------------
// 5. Decoder Layer
// ----------------------------------------------------------------------
struct Qwen2DecoderLayerImpl : torch::nn::Module {
    Qwen2RMSNorm input_layernorm{nullptr};
    Qwen2Attention self_attn{nullptr};
    Qwen2RMSNorm post_attention_layernorm{nullptr};
    Qwen2MLP mlp{nullptr};

    Qwen2DecoderLayerImpl(int64_t hidden_size, int64_t num_heads, int64_t num_key_value_heads, int64_t intermediate_size) {
        input_layernorm = register_module("input_layernorm", Qwen2RMSNorm(hidden_size));
        self_attn = register_module("self_attn", Qwen2Attention(hidden_size, num_heads, num_key_value_heads));
        post_attention_layernorm = register_module("post_attention_layernorm", Qwen2RMSNorm(hidden_size));
        mlp = register_module("mlp", Qwen2MLP(hidden_size, intermediate_size));
    }

    torch::Tensor forward(torch::Tensor hidden_states, Qwen2RotaryEmbedding& rotary_emb, torch::Tensor attention_mask = {}) {
        // Pre-LN Residual Connection
        auto residual = hidden_states;
        hidden_states = input_layernorm(hidden_states);
        hidden_states = self_attn(hidden_states, rotary_emb, attention_mask);
        hidden_states = residual + hidden_states;

        residual = hidden_states;
        hidden_states = post_attention_layernorm(hidden_states);
        hidden_states = mlp(hidden_states);
        hidden_states = residual + hidden_states;

        return hidden_states;
    }
};
TORCH_MODULE(Qwen2DecoderLayer);

// ----------------------------------------------------------------------
// 6. Full Qwen2 Model
// ----------------------------------------------------------------------
struct Qwen2ForCausalLMImpl : torch::nn::Module {
    torch::nn::Embedding embed_tokens{nullptr};
    Qwen2RotaryEmbedding rotary_emb{nullptr};
    torch::nn::ModuleList layers{nullptr};
    Qwen2RMSNorm norm{nullptr};
    torch::nn::Linear lm_head{nullptr};

    Qwen2ForCausalLMImpl(
        int64_t vocab_size = 151936,
        int64_t hidden_size = 512,
        int64_t num_hidden_layers = 4,
        int64_t num_attention_heads = 8,
        int64_t num_key_value_heads = 2,
        int64_t intermediate_size = 2048) {

        embed_tokens = register_module("embed_tokens", torch::nn::Embedding(vocab_size, hidden_size));
        rotary_emb = register_module("rotary_emb", Qwen2RotaryEmbedding(hidden_size / num_attention_heads));
        
        layers = register_module("layers", torch::nn::ModuleList());
        for (int64_t i = 0; i < num_hidden_layers; ++i) {
            layers->push_back(Qwen2DecoderLayer(hidden_size, num_attention_heads, num_key_value_heads, intermediate_size));
        }

        norm = register_module("norm", Qwen2RMSNorm(hidden_size));
        lm_head = register_module("lm_head", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, vocab_size).bias(false)));
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor attention_mask = {}) {
        auto hidden_states = embed_tokens(input_ids);

        for (size_t i = 0; i < layers->size(); ++i) {
            auto layer = layers->ptr<Qwen2DecoderLayerImpl>(i);
            hidden_states = layer->forward(hidden_states, rotary_emb, attention_mask);
        }

        hidden_states = norm(hidden_states);
        auto logits = lm_head(hidden_states);
        return logits;
    }
};
TORCH_MODULE(Qwen2ForCausalLM);

// ----------------------------------------------------------------------
// 7. 動作確認用 main 関数
// ----------------------------------------------------------------------
int main() {
    torch::manual_seed(42);

    // ハイパーパラメータの設定
    int64_t vocab_size = 151936;
    int64_t hidden_size = 512;
    int64_t num_hidden_layers = 4;
    int64_t num_attention_heads = 8;
    int64_t num_key_value_heads = 2; // GQA (Query: 8, KV: 2)
    int64_t intermediate_size = 2048;

    // モデルのインスタンス化
    Qwen2ForCausalLM model(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size);

    // ダミー入力データ (Batch Size: 2, Sequence Length: 16)
    int64_t batch_size = 2;
    int64_t seq_len = 16;
    auto input_ids = torch::randint(0, 1000, {batch_size, seq_len}, torch::kLong);

    // 順伝播の実行
    auto logits = model->forward(input_ids);

    std::cout << "Input shape  : [" << input_ids.size(0) << ", " << input_ids.size(1) << "]" << std::endl;
    std::cout << "Logits shape : [" << logits.size(0) << ", " << logits.size(1) << ", " << logits.size(2) << "]" << std::endl;

    return 0;
}