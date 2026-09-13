// 1. デバイスと型の定義
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;

// 2. 環境（Pendulum）の定義
using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, ...>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

// 3. 学習パラメータの定義
struct LOOP_CORE_PARAMETERS : rlt::rl::algorithms::td3::loop::core::DefaultParameters<...> {
    static constexpr TI STEP_LIMIT = 20000;  // 学習ステップ数
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
};

// 4. Loop Interface の構成（学習→評価→計測）
using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<...>;
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_STATE = LOOP_TIMING_CONFIG::State<LOOP_TIMING_CONFIG>;

%%writefile /content/simple_mlp.cpp

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

// シンプルな行列・ベクトル演算
struct Matrix {
    int rows, cols;
    std::vector<float> data;
    Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}
    float& operator()(int i, int j) { return data[i * cols + j]; }
    float operator()(int i, int j) const { return data[i * cols + j]; }
};

// ReLU活性化
float relu(float x) { return x > 0 ? x : 0; }

// シンプルなMLP（3層）
class SimpleMLP {
public:
    Matrix W1, b1, W2, b2, W3, b3;
    
    SimpleMLP(int input_dim, int hidden_dim, int output_dim)
        : W1(hidden_dim, input_dim), b1(hidden_dim, 1),
          W2(hidden_dim, hidden_dim), b2(hidden_dim, 1),
          W3(output_dim, hidden_dim), b3(output_dim, 1) {
        // Xavier初期化
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0, 0.1);
        auto init = [&](Matrix& m) {
            for (auto& v : m.data) v = dist(gen);
        };
        init(W1); init(W2); init(W3);
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        // 第1層
        std::vector<float> h1(W1.rows);
        for (int i = 0; i < W1.rows; i++) {
            float sum = b1(i, 0);
            for (int j = 0; j < W1.cols; j++) sum += W1(i, j) * input[j];
            h1[i] = relu(sum);
        }
        // 第2層
        std::vector<float> h2(W2.rows);
        for (int i = 0; i < W2.rows; i++) {
            float sum = b2(i, 0);
            for (int j = 0; j < W2.cols; j++) sum += W2(i, j) * h1[j];
            h2[i] = relu(sum);
        }
        // 出力層
        std::vector<float> out(W3.rows);
        for (int i = 0; i < W3.rows; i++) {
            float sum = b3(i, 0);
            for (int j = 0; j < W3.cols; j++) sum += W3(i, j) * h2[j];
            out[i] = std::tanh(sum);  // -1~1に正規化
        }
        return out;
    }
};

int main() {
    SimpleMLP actor(3, 64, 1);  // Pendulum: 観測3次元→行動1次元
    
    std::vector<float> observation = {0.5f, 0.5f, 0.1f};  // cos, sin, theta_dot
    std::vector<float> action = actor.forward(observation);
    
    std::cout << "行動: " << action[0] << std::endl;
    return 0;
}

// 5. main関数
int train() {
    DEVICE device;
    LOOP_STATE ts;
    
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    
    // 学習ループ：これだけでActor/Criticの更新、Replay Buffer管理、評価、ログ出力が全部行われる
    while (!rlt::step(device, ts)) {}
    
    rlt::free(device, ts);
    return 0;
}

