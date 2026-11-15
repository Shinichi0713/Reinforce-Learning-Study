#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>

// ============================================================================
// 1. カートポール物理環境クラス (CartPole Environment)
// ============================================================================
class CartPoleEnv {
private:
    // 物理定数
    const double gravity = 9.8;
    const double masscart = 1.0;
    const double masspole = 0.1;
    const double total_mass = masscart + masspole;
    const double length = 0.5; // ポール長さの半分
    const double polemass_length = masspole * length;
    const double force_mag = 10.0;
    const double tau = 0.02;   // 時間ステップ (秒)

    // 失敗判定の閾値
    const double x_threshold = 2.4;
    const double theta_threshold_radians = 12.0 * M_PI / 180.0;

    // 状態: [x, x_dot, theta, theta_dot]
    std::vector<double> state;
    std::mt19937 rng;

public:
    CartPoleEnv(unsigned int seed = 42) : rng(seed) {
        reset();
    }

    // 環境を初期化
    std::vector<double> reset() {
        std::uniform_real_distribution<double> dist(-0.05, 0.05);
        state = {dist(rng), dist(rng), dist(rng), dist(rng)};
        return state;
    }

    struct StepResult {
        std::vector<double> next_state;
        double reward;
        bool done;
    };

    // 1ステップ進める (action: 0=左へ押す, 1=右へ押す)
    StepResult step(int action) {
        double x = state[0];
        double x_dot = state[1];
        double theta = state[2];
        double theta_dot = state[3];

        double force = (action == 1) ? force_mag : -force_mag;
        double costheta = std::cos(theta);
        double sintheta = std::sin(theta);

        // 運動方程式による加速度計算
        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double thetaacc = (gravity * sintheta - costheta * temp) / 
                           (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        // オイラー法による状態更新
        x += tau * x_dot;
        x_dot += tau * xacc;
        theta += tau * theta_dot;
        theta_dot += tau * thetaacc;

        state = {x, x_dot, theta, theta_dot};

        // 終了条件判定
        bool done = (x < -x_threshold || x > x_threshold ||
                     theta < -theta_threshold_radians || theta > theta_threshold_radians);

        // 報酬（立っている間は毎ステップ +1.0）
        double reward = done ? 0.0 : 1.0;

        return {state, reward, done};
    }
};

// ============================================================================
// 2. Q学習エージェントクラス (Q-Learning Agent)
// ============================================================================
class QLearningAgent {
private:
    const int num_bins = 6; // 状態変数ごとの分割数
    
    // 状態空間の離散化範囲
    const double x_min = -2.4, x_max = 2.4;
    const double x_dot_min = -3.0, x_dot_max = 3.0;
    const double theta_min = -0.209, theta_max = 0.209;
    const double theta_dot_min = -3.0, theta_dot_max = 3.0;

    std::vector<double> q_table;
    int num_actions = 2;

    // ハイパーパラメータ
    double alpha = 0.1;          // 学習率
    double gamma = 0.99;         // 割引率
    double epsilon = 1.0;        // 探索率 (ε-greedy)
    double min_epsilon = 0.01;   // 探索率の下限
    double epsilon_decay = 0.995;// 探索率の減衰率

    std::mt19937 rng;

    // 連続値を離散インデックスに変換
    int digitize(double val, double min_val, double max_val) const {
        if (val <= min_val) return 0;
        if (val >= max_val) return num_bins - 1;
        double norm = (val - min_val) / (max_val - min_val);
        return static_cast<int>(norm * num_bins);
    }

    // 4次元状態から1次元配列インデックスへの変換
    int state_to_index(const std::vector<double>& state) const {
        int i1 = digitize(state[0], x_min, x_max);
        int i2 = digitize(state[1], x_dot_min, x_max);
        int i3 = digitize(state[2], theta_min, theta_max);
        int i4 = digitize(state[3], theta_dot_min, theta_dot_max);

        return i1 + num_bins * (i2 + num_bins * (i3 + num_bins * i4));
    }

public:
    QLearningAgent(unsigned int seed = 42) : rng(seed) {
        int total_states = std::pow(num_bins, 4);
        q_table.resize(total_states * num_actions, 0.0);
    }

    // 行動選択 (ε-greedy)
    int select_action(const std::vector<double>& state) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> action_dist(0, num_actions - 1);
            return action_dist(rng);
        }

        int s_idx = state_to_index(state);
        double q0 = q_table[s_idx * num_actions + 0];
        double q1 = q_table[s_idx * num_actions + 1];

        return (q0 >= q1) ? 0 : 1;
    }

    // Q値更新
    void update(const std::vector<double>& state, int action, double reward, 
                 const std::vector<double>& next_state, bool done) {
        int s_idx = state_to_index(state);
        int ns_idx = state_to_index(next_state);

        int current_pos = s_idx * num_actions + action;
        double current_q = q_table[current_pos];

        double max_next_q = 0.0;
        if (!done) {
            double nq0 = q_table[ns_idx * num_actions + 0];
            double nq1 = q_table[ns_idx * num_actions + 1];
            max_next_q = std::max(nq0, nq1);
        }

        // Q学習の更新式: Q(s,a) <- Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
        q_table[current_pos] += alpha * (reward + gamma * max_next_q - current_q);
    }

    // 探索率減衰
    void decay_epsilon() {
        epsilon = std::max(min_epsilon, epsilon * epsilon_decay);
    }
};

// ============================================================================
// 3. メイン関数
// ============================================================================
int main() {
    const int num_episodes = 1000;
    const int max_steps = 500;

    CartPoleEnv env(42);
    QLearningAgent agent(42);

    std::ofstream log_file("learning_rewards.csv");
    log_file << "Episode,Reward\n";

    std::cout << "========================================" << std::endl;
    std::cout << " CartPole Q-Learning (C++ Implementation)" << std::endl;
    std::cout << "========================================" << std::endl;

    for (int episode = 1; episode <= num_episodes; ++episode) {
        std::vector<double> state = env.reset();
        double total_reward = 0;

        for (int step = 0; step < max_steps; ++step) {
            int action = agent.select_action(state);
            auto result = env.step(action);

            agent.update(state, action, result.reward, result.next_state, result.done);

            state = result.next_state;
            total_reward += result.reward;

            if (result.done) break;
        }

        agent.decay_epsilon();
        log_file << episode << "," << total_reward << "\n";

        if (episode % 100 == 0 || episode == 1) {
            std::cout << "Episode " << std::setw(4) << episode 
                      << " | Total Reward: " << std::setw(5) << total_reward << std::endl;
        }
    }

    log_file.close();
    std::cout << "\n[Success] 学習完了！ログファイル 'learning_rewards.csv' を出力しました。" << std::endl;
    return 0;
}
