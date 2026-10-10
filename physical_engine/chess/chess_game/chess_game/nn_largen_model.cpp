#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <Eigen/Dense> // 行列演算用ライブラリの例

// 簡易的なテンソル・行列の型定義（実際にはEigenや専用テンソルライブラリを使用）
using Matrix = Eigen::MatrixXf;

class TransformerBlockCpp {
private:
    int hidden_size_;
    int num_heads_;
    int head_dim_;

    // 重み行列の例 (実際には重みファイルからロード)
    Matrix w_q_, w_k_, w_v_, w_o_;
    Matrix ffn_w1_, ffn_w2_;

    // KVキャッシュ用ストレージ（推論高速化のため前回のK, Vを保持）
    Matrix cached_k_;
    Matrix cached_v_;
    bool has_cache_ = false;

public:
    TransformerBlockCpp(int hidden_size, int num_heads) 
        : hidden_size_(hidden_size), num_heads_(num_heads) {
        head_dim_ = hidden_size_ / num_heads_;
        
        // 重みの初期化（ダミー）
        w_q_ = Matrix::Random(hidden_size_, hidden_size_);
        w_k_ = Matrix::Random(hidden_size_, hidden_size_);
        w_v_ = Matrix::Random(hidden_size_, hidden_size_);
        w_o_ = Matrix::Random(hidden_size_, hidden_size_);
        ffn_w1_ = Matrix::Random(hidden_size_, hidden_size_ * 4);
        ffn_w2_ = Matrix::Random(hidden_size_ * 4, hidden_size_);
    }

    // 順伝播とKVキャッシュ更新を含む推論メソッド
    Matrix forward(const Matrix& x, bool use_cache = true) {
        int batch_seq_len = x.rows();

        // 1. Q, K, V の線形射影
        Matrix q = x * w_q_;
        Matrix k = x * w_k_;
        Matrix v = x * w_v_;

        // 2. KVキャッシュの適用（生成AIのトークン追加生成時に再計算を防ぐ）
        if (use_cache && has_cache_) {
            // 実際の実装ではキャッシュと今回分を連結（Concatenation）する
            // ここでは簡易的にフラグ制御のみ表現
        } else if (use_cache) {
            cached_k_ = k;
            cached_v_ = v;
            has_cache_ = true;
        }

        // 3. アテンション計算の簡易表現 (Q * K^T / sqrt(d_k))
        Matrix scores = (q * k.transpose()) / std::sqrt(static_cast<float>(head_dim_));
        // Softmax等の処理（省略）

        // 4. 出力射影と残差接続
        Matrix attn_out = scores * v * w_o_;
        Matrix x_norm = x + attn_out; // 残差接続の簡略化

        // 5. FFN (Feed-Forward Network)
        Matrix ffn_hidden = (x_norm * ffn_w1_).cwiseMax(0.0f); // GELU/ReLUの代わりのReLU活性化
        Matrix ffn_out = ffn_hidden * ffn_w2_;
        
        return x_norm + ffn_out;
    }

    void clear_cache() {
        has_cache_ = false;
        cached_k_.resize(0, 0);
        cached_v_.resize(0, 0);
    }
};

// --- 実行テスト ---
int main() {
    int hidden_size = 64;
    int num_heads = 4;

    TransformerBlockCpp block(hidden_size, num_heads);

    // ダミー入力 (シーケンス長 1, 隠れ次元 64)
    Matrix input_tensor = Matrix::Random(1, hidden_size);

    // 推論の実行
    Matrix output = block.forward(input_tensor, true);

    std::cout << "C++ Transformer ブロックの推論が完了しました。" << std::endl;
    std::cout << "出力行列の形状: " << output.rows() << " x " << output.cols() << std::endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf;

/**
 * @brief 1層分のK/Vキャッシュを管理するクラス
 */
class KVCache {
private:
    Matrix key_cache_;
    Matrix value_cache_;
    bool is_initialized_;

public:
    KVCache() : is_initialized_(false) {}

    /**
     * @brief 新しいトークンのK, Vを受け取り、既存のキャッシュと結合（Concatenate）する
     * @param new_k 今回のKey行列 (batch_size * new_seq_len, head_dim) ※簡略化のため2D
     * @param new_v 今回のValue行列 (batch_size * new_seq_len, head_dim)
     */
    void update(const Matrix& new_k, const Matrix& new_v) {
        if (!is_initialized_) {
            // 初回（プロンプト処理時）はそのままキャッシュに格納
            key_cache_ = new_k;
            value_cache_ = new_v;
            is_initialized_ = true;
        } else {
            // 2回目以降（1トークンずつの逐次生成時）は末尾に結合
            key_cache_ = concat_rows(key_cache_, new_k);
            value_cache_ = concat_rows(value_cache_, new_v);
        }
    }

    // キャッシュされたKeyを取得
    const Matrix& get_key() const {
        return key_cache_;
    }

    // キャッシュされたValueを取得
    const Matrix& get_value() const {
        return value_cache_;
    }

    // キャッシュのリセット（新しい対話セッション開始時など）
    void clear() {
        is_initialized_ = false;
        key_cache_.resize(0, 0);
        value_cache_.resize(0, 0);
    }

    bool empty() const {
        return !is_initialized_;
    }

private:
    // 行方向（シーケンス方向）への行列結合ヘルパー
    Matrix concat_rows(const Matrix& a, const Matrix& b) {
        Matrix result(a.rows() + b.rows(), a.cols());
        result.topRows(a.rows()) = a;
        result.bottomRows(b.rows()) = b;
        return result;
    }
};

// --- 動作確認用メイン関数 ---
int main() {
    int head_dim = 16;
    KVCache layer_cache;

    std::cout << "=== KVキャッシュの動作テスト ===" << std::endl;

    // 1. プレフィックス（プロンプト）入力時: シーケンス長 3 のK, Vが渡された場合
    Matrix prompt_k = Matrix::Random(3, head_dim);
    Matrix prompt_v = Matrix::Random(3, head_dim);

    layer_cache.update(prompt_k, prompt_v);
    std::cout << "1. プロンプト処理後のキャッシュ形状 (Key): " 
              << layer_cache.get_key().rows() << " x " << layer_cache.get_key().cols() << std::endl;

    // 2. 1トークン目生成時: シーケンス長 1 の新しいK, Vが渡された場合
    Matrix token1_k = Matrix::Random(1, head_dim);
    Matrix token1_v = Matrix::Random(1, head_dim);

    layer_cache.update(token1_k, token1_v);
    std::cout << "2. 1トークン生成追加後のキャッシュ形状 (Key): " 
              << layer_cache.get_key().rows() << " x " << layer_cache.get_key().cols() << std::endl;

    // 3. 2トークン目生成時
    Matrix token2_k = Matrix::Random(1, head_dim);
    Matrix token2_v = Matrix::Random(1, head_dim);

    layer_cache.update(token2_k, token2_v);
    std::cout << "3. 2トークン生成追加後のキャッシュ形状 (Key): " 
              << layer_cache.get_key().rows() << " x " << layer_cache.get_key().cols() << std::endl;

    return 0;
}