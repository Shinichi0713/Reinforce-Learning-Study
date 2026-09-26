#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::RowVectorXf;

const int BOARD_SIZE = 19;
const int ACTION_SIZE = BOARD_SIZE * BOARD_SIZE; // 361
const int INPUT_SIZE = 3 * BOARD_SIZE * BOARD_SIZE; // 3チャネル (自分, 相手, 手番)

// --- 1. ソフトマックス付き全結合ネットワーク（Policy Network）---
class PolicyNetwork {
private:
    Matrix W1, W2;
    Vector b1, b2;

public:
    PolicyNetwork(int input_dim, int hidden_dim, int output_dim) {
        std::random_device rd;
        std::mt19937 gen(rd());
        // Xavier/He初期化
        float std1 = std::sqrt(2.0f / input_dim);
        float std2 = std::sqrt(2.0f / hidden_dim);
        std::normal_distribution<float> d1(0.0f, std1);
        std::normal_distribution<float> d2(0.0f, std2);

        W1 = Matrix(input_dim, hidden_dim);
        for (int i = 0; i < input_dim; ++i)
            for (int j = 0; j < hidden_dim; ++j) W1(i, j) = d1(gen);

        W2 = Matrix(hidden_dim, output_dim);
        for (int i = 0; i < hidden_dim; ++i)
            for (int j = 0; j < output_dim; ++j) W2(i, j) = d2(gen);

        b1 = Vector::Zero(hidden_dim);
        b2 = Vector::Zero(output_dim);
    }

    // 順伝播：状態ベクトルから各行動の確率分布（Softmax）を計算
    struct ForwardResult {
        Vector h1;     // 隠れ層出力 (ReLU適用後)
        Vector logits; // 出力層ロジット
        Vector probs;  // 確率分布 (Softmax)
    };

    ForwardResult forward(const Vector& x, const std::vector<bool>& legal_mask) {
        // 1. 隠れ層 (Linear + ReLU)
        Vector h1 = (x * W1 + b1).unaryExpr([](float val) { return std::max(0.0f, val); });

        // 2. 出力層 (Linear)
        Vector logits = h1 * W2 + b2;

        // 3. 非合法手のマスキング (-1e9 を加算して確率を0にする)
        for (int i = 0; i < ACTION_SIZE; ++i) {
            if (!legal_mask[i]) {
                logits(i) = -1e9f;
            }
        }

        // 4. Softmax 計算 (数値的安定性のために max を引く)
        float max_logit = logits.maxCoeff();
        Vector exp_logits = (logits.array() - max_logit).exp();
        float sum_exp = exp_logits.sum();
        Vector probs = exp_logits / sum_exp;

        return {h1, logits, probs};
    }

    // パラメータ更新 (方策勾配法: Policy Gradient)
    // 損失の勾配 dL/dLogits = (probs - 1_at_action) * G_t
    void update(const Vector& x, const Vector& h1, const Vector& probs, int chosen_action, float G_t, float lr) {
        // 1. 出力層の勾配 dL/dLogits
        Vector d_logits = probs;
        d_logits(chosen_action) -= 1.0f;
        d_logits *= G_t; // 累積報酬 (Return) によるスケーリング

        // 2. 重みとバイアスの勾配計算
        Matrix d_W2 = h1.transpose() * d_logits;
        Vector d_b2 = d_logits;

        // 3. 隠れ層への逆伝播
        Vector d_h1 = d_logits * W2.transpose();
        // ReLU の微分
        Vector d_relu = h1.unaryExpr([](float val) { return val > 0.0f ? 1.0f : 0.0f; });
        Vector d_hidden = d_h1.cwiseProduct(d_relu);

        Matrix d_W1 = x.transpose() * d_hidden;
        Vector d_b1 = d_hidden;

        // 4. SGD パラメータ更新
        W2 -= lr * d_W2;
        b2 -= lr * d_b2;
        W1 -= lr * d_W1;
        b1 -= lr * d_b1;
    }
};

// --- 2. 確率分布からのアクションサンプリング ---
int sample_action(const Vector& probs) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    float r = dis(gen);
    float cumsum = 0.0f;
    for (int i = 0; i < ACTION_SIZE; ++i) {
        cumsum += probs(i);
        if (r <= cumsum) return i;
    }
    return ACTION_SIZE - 1;
}

// --- 3. 1エピソードの記憶保持用構造体 ---
struct StepMemory {
    Vector state;
    Vector hidden_state;
    Vector probs;
    int action;
    float reward;
};

// --- 4. メイン学習ルーチン ---
int main() {
    // 状態入力(3*19*19=1083) -> 隠れ層(128) -> 行動空間(361)
    PolicyNetwork policy_net(INPUT_SIZE, 128, ACTION_SIZE);

    float lr = 0.001f;
    float gamma = 0.99f; // 割引率
    int num_episodes = 500;

    std::cout << "--- C++ 方策勾配法 (REINFORCE) 学習開始 ---" << std::endl;

    for (int episode = 1; episode <= num_episodes; ++episode) {
        std::vector<StepMemory> episode_memory;
        
        // 仮のダミー環境ループ（実際の Raylib GoGame インスタンスと置き換えます）
        bool done = false;
        int step_count = 0;
        
        while (!done && step_count < 50) { // 例として1エピソード50手
            // 本来は Raylib GoGame の盤面状態から 1x1083 テンソルを作成
            Vector current_state = Vector::Random(INPUT_SIZE);
            
            // 合法手マスク (ダミー: すべての着手を許可)
            std::vector<bool> legal_mask(ACTION_SIZE, true);

            // 1. 順伝播
            auto res = policy_net.forward(current_state, legal_mask);

            // 2. アクション選択
            int action = sample_action(res.probs);

            // 3. 環境から報酬を取得（ダミー報酬）
            float reward = (step_count == 49) ? 1.0f : 0.0f; // 終局時に勝利報酬+1.0

            episode_memory.push_back({current_state, res.h1, res.probs, action, reward});

            step_count++;
            if (step_count >= 50) done = true;
        }

        // --- 4. エピソード終了後のバックプロパゲーション (REINFORCE) ---
        float G = 0.0f; // 累積割引報酬 (Return)
        
        // 後ろの手から遡って勾配を計算・更新
        for (int t = static_cast<int>(episode_memory.size()) - 1; t >= 0; --t) {
            const auto& mem = episode_memory[t];
            G = mem.reward + gamma * G; // G_t = R_t + gamma * G_{t+1}

            // 方策のパラメータを累積報酬 G で更新
            policy_net.update(mem.state, mem.hidden_state, mem.probs, mem.action, G, lr);
        }

        if (episode % 50 == 0) {
            std::cout << "Episode " << episode << " 完了 | 終局時報酬 G_0: " << G << std::endl;
        }
    }

    std::cout << "学習完了。" << std::endl;
    return 0;
}