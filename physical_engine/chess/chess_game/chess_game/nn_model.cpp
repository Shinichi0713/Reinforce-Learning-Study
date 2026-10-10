#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

using Matrix = std::vector<std::vector<float>>;
using Tensor3D = std::vector<std::vector<std::vector<float>>>; // [Height, Width, Channels]

// 行列の型定義 (2次元 vector)
using Matrix = std::vector<std::vector<float>>;
using Matrix = Eigen::MatrixXf;
using Vector = Eigen::RowVectorXf;

// --- 1. レイヤー基底クラス (Rustの Trait に相当) ---
class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& output_gradient, float lr) = 0;
};

// --- 2. 全結合層 (Dense Layer) ---
class Dense : public Layer {
private:
    Matrix weights;
    Vector biases;
    Matrix input_cache;

public:
    Dense(int in_features, int out_features) {
        // He初期化 (Kaiming Normal)
        std::random_device rd;
        std::mt19937 gen(rd());
        float std_dev = std::sqrt(2.0f / static_cast<float>(in_features));
        std::normal_distribution<float> d(0.0f, std_dev);

        weights = Matrix(in_features, out_features);
        for (int i = 0; i < in_features; ++i) {
            for (int j = 0; j < out_features; ++j) {
                weights(i, j) = d(gen);
            }
        }
        biases = Vector::Zero(out_features);
    }

    Matrix forward(const Matrix& input) override {
        input_cache = input;
        // Y = X * W + b (ブロードキャスト加算)
        return (input * weights).rowwise() + biases;
    }

    Matrix backward(const Matrix& output_gradient, float lr) override {
        // 勾配計算
        // dL/dW = X^T * dL/dY
        Matrix d_weights = input_cache.transpose() * output_gradient;
        // dL/db = sum(dL/dY, axis=0)
        Vector d_biases = output_gradient.colwise().sum();
        // dL/dX = dL/dY * W^T
        Matrix d_input = output_gradient * weights.transpose();

        // パラメータ更新 (SGD)
        weights -= d_weights * lr;
        biases -= d_biases * lr;

        return d_input;
    }
};

// --- 3. ReLU 活性化層 ---
class ReLU : public Layer {
private:
    Matrix output_cache;

public:
    Matrix forward(const Matrix& input) override {
        output_cache = input.unaryExpr([](float x) { return std::max(0.0f, x); });
        return output_cache;
    }

    Matrix backward(const Matrix& output_gradient, float lr) override {
        // ReLUの微分: 出力が > 0 なら 1.0, それ以外は 0.0
        Matrix d_relu = output_cache.unaryExpr([](float x) { return x > 0.0f ? 1.0f : 0.0f; });
        return output_gradient.cwiseProduct(d_relu);
    }
};

// --- 4. Sigmoid 活性化層 ---
class Sigmoid : public Layer {
private:
    Matrix output_cache;

public:
    Matrix forward(const Matrix& input) override {
        output_cache = input.unaryExpr([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
        return output_cache;
    }

    Matrix backward(const Matrix& output_gradient, float lr) override {
        // Sigmoidの微分: sig * (1 - sig)
        Matrix d_sigmoid = output_cache.binaryExpr(
            Matrix::Ones(output_cache.rows(), output_cache.cols()),
            [](float out, float one) { return out * (one - out); }
        );
        return output_gradient.cwiseProduct(d_sigmoid);
    }
};

// --- 5. 積層モデル (Sequential) ---
class Sequential {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Matrix forward(Matrix x) {
        for (auto& layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }

    void backward(Matrix gradient, float lr) {
        // 逆伝播（後ろのレイヤーから順に処理）
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradient = (*it)->backward(gradient, lr);
        }
    }
};

// --- 6. メイン実行コード (XOR問題の学習) ---
int main() {
    // 2入力 -> 8隠れ層(ReLU) -> 4隠れ層(ReLU) -> 1出力(Sigmoid) のモデル構築
    Sequential model;
    model.add(std::make_unique<Dense>(2, 8));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(8, 4));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dense>(4, 1));
    model.add(std::make_unique<Sigmoid>());

    // XOR問題のデータセット
    Matrix X(4, 2);
    X << 0.0f, 0.0f,
         0.0f, 1.0f,
         1.0f, 0.0f,
         1.0f, 1.0f;

    Matrix Y(4, 1);
    Y << 0.0f,
         1.0f,
         1.0f,
         0.0f;

    int epochs = 10000;
    float lr = 0.1f;

    std::cout << "--- 学習開始 ---" << std::endl;
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        // 1. 順伝播
        Matrix predictions = model.forward(X);

        // 2. 損失計算 (MSE)
        Matrix diff = predictions - Y;
        float loss = diff.array().square().mean();

        // 3. 損失の微分 (dL/dPred = 2 * (pred - y) / N)
        Matrix loss_gradient = 2.0f * diff / static_cast<float>(X.rows());

        // 4. 逆伝播による更新
        model.backward(loss_gradient, lr);

        if (epoch % 2000 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss << std::endl;
        }
    }

    std::cout << "\n--- 学習完了後の予測結果 ---" << std::endl;
    Matrix final_preds = model.forward(X);
    for (int i = 0; i < X.rows(); ++i) {
        std::cout << "Input: [" << X(i, 0) << ", " << X(i, 1) << "]"
                  << " -> Pred: " << final_preds(i, 0)
                  << " (Target: " << Y(i, 0) << ")" << std::endl;
    }

    return 0;
}




// ----------------------------------------------------------------------
// 1. 行列の掛け算: A (M x K) * B (K x N) -> C (M x N)
// ----------------------------------------------------------------------
Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B[0].size();

    Matrix C(M, std::vector<float>(N, 0.0f));
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// ----------------------------------------------------------------------
// 2. 行列の転置: A (M x N) -> A^T (N x M)
// ----------------------------------------------------------------------
Matrix transpose(const Matrix& A) {
    size_t M = A.size();
    size_t N = A[0].size();

    Matrix At(N, std::vector<float>(M));
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            At[j][i] = A[i][j];
        }
    }
    return At;
}

// ----------------------------------------------------------------------
// 3. 数値的に安定な Softmax (行ごとに適用)
// ----------------------------------------------------------------------
Matrix softmax(const Matrix& input) {
    Matrix output = input;
    for (size_t i = 0; i < input.size(); ++i) {
        // オーバーフロー防止のため行内の最大値を抽出
        float max_val = *std::max_element(input[i].begin(), input[i].end());
        
        float sum = 0.0f;
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] = std::exp(input[i][j] - max_val);
            sum += output[i][j];
        }
        
        // 確率の和が 1 になるよう正規化
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] /= sum;
        }
    }
    return output;
}

// ----------------------------------------------------------------------
// 4. Scaled Dot-Product Attention の本体
// ----------------------------------------------------------------------
Matrix scaled_dot_product_attention(
    const Matrix& Q, // [Seq_Len, D_k]
    const Matrix& K, // [Seq_Len, D_k]
    const Matrix& V  // [Seq_Len, D_v]
) {
    size_t d_k = Q[0].size();
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    // Step 1: Q * K^T の計算 [Seq_Len, Seq_Len]
    Matrix K_T = transpose(K);
    Matrix scores = matmul(Q, K_T);

    // Step 2: スケーリング (sqrt(d_k) で割る)
    for (size_t i = 0; i < scores.size(); ++i) {
        for (size_t j = 0; j < scores[0].size(); ++j) {
            scores[i][j] *= scale;
        }
    }

    // Step 3: Softmax (アテンションウェイトの算出)
    Matrix attn_weights = softmax(scores);

    // Step 4: Attention Weights * V の計算 [Seq_Len, D_v]
    Matrix output = matmul(attn_weights, V);

    return output;
}

// ユーティリティ: 行列の表示
void print_matrix(const std::string& name, const Matrix& mat) {
    std::cout << "=== " << name << " ===" << std::endl;
    for (const auto& row : mat) {
        for (float val : row) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// ----------------------------------------------------------------------
// 5. 動作確認用 main 関数
// ----------------------------------------------------------------------
int main() {
    // シーケンス長 = 3, 次元数 d_k = 4, d_v = 4 のダミーデータ
    Matrix Q = {
        {1.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 2.0f, 0.0f, 2.0f},
        {1.0f, 1.0f, 0.0f, 0.0f}
    };

    Matrix K = {
        {1.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f, 1.0f}
    };

    Matrix V = {
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.5f, 0.6f, 0.7f, 0.8f},
        {0.9f, 1.0f, 1.1f, 1.2f}
    };

    print_matrix("Query (Q)", Q);
    print_matrix("Key (K)", K);
    print_matrix("Value (V)", V);

    // アテンション計算の実行
    Matrix output = scaled_dot_product_attention(Q, K, V);

    print_matrix("Attention Output", output);

    return 0;
}



// ----------------------------------------------------------------------
// 1. 基本的な線形代数 & 数学関数
// ----------------------------------------------------------------------
Matrix matmul(const Matrix& A, const Matrix& B) {
    size_t M = A.size(), K = A[0].size(), N = B[0].size();
    Matrix C(M, std::vector<float>(N, 0.0f));
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            for (size_t j = 0; j < N; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    size_t M = A.size(), N = A[0].size();
    Matrix At(N, std::vector<float>(M));
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            At[j][i] = A[i][j];
    return At;
}

Matrix softmax(const Matrix& input) {
    Matrix output = input;
    for (size_t i = 0; i < input.size(); ++i) {
        float max_val = *std::max_element(input[i].begin(), input[i].end());
        float sum = 0.0f;
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] = std::exp(input[i][j] - max_val);
            sum += output[i][j];
        }
        for (size_t j = 0; j < input[i].size(); ++j) {
            output[i][j] /= sum;
        }
    }
    return output;
}

// ----------------------------------------------------------------------
// 2. 簡易 CNN 演算 (2D Convolution)
// ----------------------------------------------------------------------
Tensor3D conv2d_simple(const Tensor3D& input, size_t out_channels) {
    size_t in_h = input.size();
    size_t in_w = input[0].size();
    size_t in_c = input[0][0].size();

    // 3x3 フィルタ, Stride=1, Padding=1 (サイズ変更なし)
    size_t out_h = in_h;
    size_t out_w = in_w;

    Tensor3D output(out_h, std::vector<std::vector<float>>(out_w, std::vector<float>(out_channels, 0.0f)));

    // ダミーの畳み込み重み（固定値で模擬）
    for (size_t h = 0; h < out_h; ++h) {
        for (size_t w = 0; w < out_w; ++w) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                float val = 0.0f;
                // 周辺3x3のカーネル演算
                for (int dh = -1; dh <= 1; ++dh) {
                    for (int dw = -1; dw <= 1; ++dw) {
                        int ih = static_cast<int>(h) + dh;
                        int iw = static_cast<int>(w) + dw;
                        if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w)) {
                            for (size_t ic = 0; ic < in_c; ++ic) {
                                float weight = 0.1f * (oc + 1) * (dh + dw + 2);
                                val += input[ih][iw][ic] * weight;
                            }
                        }
                    }
                }
                // ReLU アクティベーション
                output[h][w][oc] = std::max(0.0f, val);
            }
        }
    }
    return output;
}

// ----------------------------------------------------------------------
// 3. CNN 特徴マップからトークン系列への変換
// ----------------------------------------------------------------------
// [H, W, C] -> [H * W, C] (空間グリッドをトークン化)
Matrix flatten_feature_map(const Tensor3D& feat_map) {
    size_t H = feat_map.size();
    size_t W = feat_map[0].size();
    size_t C = feat_map[0][0].size();

    Matrix tokens(H * W, std::vector<float>(C, 0.0f));
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            size_t token_idx = h * W + w;
            for (size_t c = 0; c < C; ++c) {
                tokens[token_idx][c] = feat_map[h][w][c];
            }
        }
    }
    return tokens;
}

// ----------------------------------------------------------------------
// 4. Scaled Dot-Product Attention (空間アテンション)
// ----------------------------------------------------------------------
std::pair<Matrix, Matrix> spatial_attention(const Matrix& X) {
    // X: [N, C]  (N = H * W トークン数, C = チャネル数)
    size_t N = X.size();
    size_t C = X[0].size();

    // Query, Key, Value への射影（手動での重み乗算）
    Matrix Q = X;
    Matrix K = X;
    Matrix V = X;

    // Q * K^T の計算 -> [N, N] (各領域同士の関心度)
    Matrix K_T = transpose(K);
    Matrix scores = matmul(Q, K_T);

    float scale = 1.0f / std::sqrt(static_cast<float>(C));
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            scores[i][j] *= scale;
        }
    }

    // Softmax でアテンションマップ生成
    Matrix attn_weights = softmax(scores);

    // アテンション重みを Value に適用 -> [N, C]
    Matrix attn_output = matmul(attn_weights, V);

    return {attn_output, attn_weights};
}

// ----------------------------------------------------------------------
// 5. CNN + Attention 統合モデルによる推論パイプライン
// ----------------------------------------------------------------------
void run_cnn_attention_pipeline() {
    std::cout << "=== 1. 入力画像の準備 (H=4, W=4, C=3) ===" << std::endl;
    size_t H = 4, W = 4, C = 3;
    Tensor3D image(H, std::vector<std::vector<float>>(W, std::vector<float>(C, 0.0f)));

    // 中心付近 (h=2, w=2) に高い画素値（特定オブジェクト）を配置
    image[2][2] = {1.0f, 0.8f, 0.9f};
    image[2][1] = {0.7f, 0.6f, 0.5f};

    std::cout << "=== 2. CNN による空間特徴量の抽出 ===" << std::endl;
    size_t cnn_out_channels = 8;
    Tensor3D feature_map = conv2d_simple(image, cnn_out_channels);
    std::cout << "CNN Feature Map Shape: [" << H << ", " << W << ", " << cnn_out_channels << "]\n\n";

    std::cout << "=== 3. 特徴マップのトークン系列化 (Flatten) ===" << std::endl;
    Matrix tokens = flatten_feature_map(feature_map); // [16, 8]
    std::cout << "Tokens Shape: [" << tokens.size() << ", " << tokens[0].size() << "]\n\n";

    std::cout << "=== 4. 空間アテンション機構の適用 ===" << std::endl;
    auto [attn_out, attn_weights] = spatial_attention(tokens);

    std::cout << "Attention Weights Matrix [16 x 16] (部分表示):\n";
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            std::cout << std::setw(7) << std::fixed << std::setprecision(3) << attn_weights[i][j] << " ";
        }
        std::cout << "...\n";
    }
    std::cout << "...\n\n";

    std::cout << "=== 5. グローバル集約 & 分類 (Linear Classifier) ===" << std::endl;
    // トークン方向に平均プール（Global Average Pooling）して1つの特徴ベクトルに縮約
    std::vector<float> global_vector(cnn_out_channels, 0.0f);
    for (size_t i = 0; i < attn_out.size(); ++i) {
        for (size_t c = 0; c < cnn_out_channels; ++c) {
            global_vector[c] += attn_out[i][c] / attn_out.size();
        }
    }

    // クラス出力層 (2クラス分類の Logits を計算)
    Matrix W_cls = {
        {0.5f, -0.2f, 0.1f, 0.8f, -0.5f, 0.3f, 0.1f, -0.4f}, // Class 0
        {-0.3f, 0.6f, 0.4f, -0.1f, 0.9f, -0.2f, 0.5f, 0.7f}   // Class 1
    };

    std::vector<float> logits(2, 0.0f);
    for (size_t cls = 0; cls < 2; ++cls) {
        for (size_t c = 0; c < cnn_out_channels; ++c) {
            logits[cls] += W_cls[cls][c] * global_vector[c];
        }
    }

    std::cout << "Logits (Class 0 / Class 1): " << logits[0] << " / " << logits[1] << std::endl;
    int predicted_class = (logits[1] > logits[0]) ? 1 : 0;
    std::cout << ">>> 最終予測クラス: " << predicted_class << " <<<\n";
}

int main() {
    run_cnn_attention_pipeline();
    return 0;
}