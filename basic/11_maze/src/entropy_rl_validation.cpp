#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>

// ============================================================
// 強化学習における「エントロピー正則化」の効果を検証するC++コード
// ============================================================
// ・2つの行動（行動0: 右, 行動1: 左）を持つ方策を想定
// ・Softmax方策のロジットを勾法で更新
// ・「エントロピー項なし」と「エントロピー項あり」の2パターンを比較
// ・エントロピーが探索に与える影響を数値的に確認する
// ============================================================

const int NUM_ACTIONS = 2;
const double LR = 0.1;          // 学習率
const double H = 1e-5;           // 数値微分用の微小量
const int ITERATIONS = 100;      // 更新回数

// 各行動の真の報酬（行動0の方が高いが、行動1も悪くない）
const std::vector<double> TRUE_REWARDS = {1.0, 0.6};

// Softmax関数: logits -> 確率
std::vector<double> softmax(const std::vector<double>& logits) {
    double max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    std::vector<double> probs(NUM_ACTIONS);

    for (int i = 0; i < NUM_ACTIONS; ++i) {
        probs[i] = std::exp(logits[i] - max_logit); // オーバーフロー対策
        sum_exp += probs[i];
    }
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        probs[i] /= sum_exp;
    }
    return probs;
}

// シャノンエントロピー: H = -sum( p_i * log(p_i) )
double compute_entropy(const std::vector<double>& probs) {
    double entropy = 0.0;
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        if (probs[i] > 1e-12) {
            entropy -= probs[i] * std::log(probs[i]);
        }
    }
    return entropy;
}

// 期待報酬: sum( p_i * r_i )
double expected_reward(const std::vector<double>& probs) {
    double reward = 0.0;
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        reward += probs[i] * TRUE_REWARDS[i];
    }
    return reward;
}

// 損失関数: L = -E[R] - c_entropy * H(π)
// （最小化したいので、報酬は負号を付ける。エントロピーは正則化項として引く）
double loss_function(const std::vector<double>& logits, double c_entropy) {
    std::vector<double> probs = softmax(logits);
    double exp_reward = expected_reward(probs);
    double entropy = compute_entropy(probs);
    return -exp_reward - c_entropy * entropy;
}

// 数値微分（中心差分）による勾配計
std::vector<double> compute_gradients(const std::vector<double>& logits, double c_entropy) {
    std::vector<double> grads(NUM_ACTIONS);
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        std::vector<double> logits_plus = logits;
        std::vector<double> logits_minus = logits;
        logits_plus[i] += H;
        logits_minus[i] -= H;
        grads[i] = (loss_function(logits_plus, c_entropy) - loss_function(logits_minus, c_entropy)) / (2.0 * H);
    }
    return grads;
}

// 1ステップの更新（勾配降下）
void update(std::vector<double>& logits, double c_entropy) {
    std::vector<double> grads = compute_gradients(logits, c_entropy);
    for (int i = 0; i < NUM_ACTIONS; ++i) {
        logits[i] -= LR * grads[i];
    }
}

// シミュレーションを実行して結果を出力
void run_simulation(double c_entropy, const std::string& label) {
    std::vector<double> logits(NUM_ACTIONS, 0.0); // 初期値: [0.0, 0.0]（一様）

    std::cout << "\n=== " << label << " (c_entropy = " << c_entropy << ") ===" << std::endl;
    std::cout << std::setw(5) << "Iter" 
              << std::setw(14) << "π(右)" 
              << std::setw(14) << "π(左)" 
              << std::setw(14) << "Entropy" 
              << std::setw(18) << "E[Reward]" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int iter = 0; iter <= ITERATIONS; ++iter) {
        std::vector<double> probs = softmax(logits);
        double entropy = compute_entropy(probs);
        double exp_r = expected_reward(probs);

        if (iter % 10 == 0 || iter == ITERATIONS) {
            std::cout << std::fixed << std::setprecision(6)
                      << std::setw(5) << iter
                      << std::setw(14) << probs[0]
                      << std::setw(14) << probs[1]
                      << std::setw(14) << entropy
                      << std::setw(18) << exp_r << std::endl;
        }

        if (iter < ITERATIONS) {
            update(logits, c_entropy);
        }
    }
}

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  強化学習: エントロピー正則化の効果検証プログラム" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "設定:" << std::endl;
    std::cout << "  行動0（右）の報酬: " << TRUE_REWARDS[0] << std::endl;
    std::cout << "  行動1（左）の報酬: " << TRUE_REWARDS[1] << std::endl;
    std::cout << "  学習率: " << LR << std::endl;
    std::cout << "  更新回数: " << ITERATIONS << std::endl;

    // ケース1: エントロピー項なし（c_entropy = 0）
    // → 報酬の高い行動0に確率が集中し、エントロピーが0に近づく（局所最適・探索停止）
    run_simulation(0.0, "エントロピー項なし");

    // ケース2: エントロピー項あり（c_entropy = 0.3）
    // → 報酬を最大化しつつ、行動の多様性（エントロピー）を維持する
    run_simulation(0.3, "エントロピー項あり");

    std::cout << "\n============================================================" << std::endl;
    std::cout << "【検証結果の解説】" << std::endl;
    std::cout << "  1. エントロピー項なし: π(右) -> 1.0 に近づき、Entropy -> 0" << std::endl;
    std::cout << "     → 一方の行動に固執し、他の行動を試さなくなる（探索停止）" << std::endl;
    std::cout << "  2. エントロピー項あり: π(右) > π(左) だが、Entropy > 0 を維持" << std::endl;
    std::cout << "     → 報酬の高い行動を優先しつつ、他の行動の可能性も残す（探索維持）" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
