#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>

// --- 1. Q-Network の定義 ---
struct QNetworkImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    QNetworkImpl(int64_t state_dim, int64_t action_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, action_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};
TORCH_MODULE(QNetwork);

// --- 2. Replay Buffer の定義 ---
struct Experience {
    torch::Tensor state;
    int64_t action;
    float reward;
    torch::Tensor next_state;
    bool done;
};

class ReplayBuffer {
private:
    std::deque<Experience> buffer;
    size_t capacity;
    std::mt19937 rng{std::random_device{}()};

public:
    explicit ReplayBuffer(size_t cap) : capacity(cap) {}

    void push(const torch::Tensor& state, int64_t action, float reward, const torch::Tensor& next_state, bool done) {
        if (buffer.size() >= capacity) {
            buffer.pop_front();
        }
        buffer.push_back({state, action, reward, next_state, done});
    }

    std::vector<Experience> sample(size_t batch_size) {
        std::vector<Experience> result;
        std::vector<size_t> indices(buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t i = 0; i < std::min(batch_size, buffer.size()); ++i) {
            result.push_back(buffer[indices[i]]);
        }
        return result;
    }

    size_t size() const { return buffer.size(); }
};

// --- 3. 簡略化されたダミー環境 (CartPole風) ---
class DummyEnv {
private:
    torch::Tensor state;
    int steps;

public:
    DummyEnv() { reset(); }

    torch::Tensor reset() {
        state = torch::randn({4}); // 4次元状態
        steps = 0;
        return state;
    }

    std::tuple<torch::Tensor, float, bool> step(int64_t action) {
        steps++;
        // 次状態の遷移ロジック（ダミー）
        torch::Tensor next_state = state + torch::randn({4}) * 0.1 + (action == 1 ? 0.05 : -0.05);
        state = next_state;

        float reward = 1.0f;
        bool done = (steps >= 200) || (torch::abs(state[0]).item<float>() > 2.4f);

        return {next_state, reward, done};
    }
};

// --- 4. DQN エージェントの定義 ---
class DQNAgent {
private:
    int64_t state_dim;
    int64_t action_dim;
    QNetwork q_net{nullptr};
    QNetwork target_net{nullptr};
    torch::optim::Adam optimizer;
    ReplayBuffer buffer;

    float gamma = 0.99f;
    float epsilon = 1.0f;
    float epsilon_min = 0.01f;
    float epsilon_decay = 0.995f;
    size_t batch_size = 64;

    std::mt19937 rng{std::random_device{}()};

public:
    DQNAgent(int64_t s_dim, int64_t a_dim, size_t buffer_cap)
        : state_dim(s_dim), action_dim(a_dim),
          q_net(s_dim, a_dim), target_net(s_dim, a_dim),
          optimizer(q_net->parameters(), torch::optim::AdamOptions(1e-3)),
          buffer(buffer_cap) {
        
        // Target Networkにパラメータを同期
        update_target_network();
    }

    int64_t select_action(const torch::Tensor& state) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int64_t> action_dist(0, action_dim - 1);
            return action_dist(rng);
        } else {
            torch::NoGradGuard no_grad;
            torch::Tensor q_values = q_net->forward(state.unsqueeze(0));
            return q_values.argmax(1).item<int64_t>();
        }
    }

    void remember(const torch::Tensor& s, int64_t a, float r, const torch::Tensor& ns, bool d) {
        buffer.push(s, a, r, ns, d);
    }

    void train_step() {
        if (buffer.size() < batch_size) return;

        auto batch = buffer.sample(batch_size);

        // バッチテンソルの作成
        std::vector<torch::Tensor> states, next_states, rewards, dones, actions;
        for (const auto& exp : batch) {
            states.push_back(exp.state);
            next_states.push_back(exp.next_state);
            rewards.push_back(torch::tensor({exp.reward}));
            dones.push_back(torch::tensor({exp.done ? 1.0f : 0.0f}));
            actions.push_back(torch::tensor({exp.action}));
        }

        torch::Tensor states_t = torch::stack(states);
        torch::Tensor next_states_t = torch::stack(next_states);
        torch::Tensor rewards_t = torch::stack(rewards);
        torch::Tensor dones_t = torch::stack(dones);
        torch::Tensor actions_t = torch::stack(actions);

        // 現在のQ値 Q(s, a) の取得
        torch::Tensor q_values = q_net->forward(states_t).gather(1, actions_t);

        // ターゲットQ値の計算: r + gamma * max_a Q_target(s', a) * (1 - done)
        torch::Tensor next_q_values;
        {
            torch::NoGradGuard no_grad;
            next_q_values = std::get<0>(target_net->forward(next_states_t).max(1, true));
        }
        torch::Tensor target_q = rewards_t + (gamma * next_q_values * (1.0f - dones_t));

        // Smooth L1 Loss (Huber Loss) で誤差計算
        torch::Tensor loss = torch::smooth_l1_loss(q_values, target_q);

        // 勾配更新
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // epsilonの減衰
        if (epsilon > epsilon_min) {
            epsilon *= epsilon_decay;
        }
    }

    void update_target_network() {
        torch::NoGradGuard no_grad;
        auto target_params = target_net->named_parameters();
        auto q_params = q_net->named_parameters();
        
        for (auto& param : target_params) {
            param.value().copy_(q_params[param.key()]);
        }
    }
};

// --- 5. メイン学習ループ ---
int main() {
    constexpr int64_t STATE_DIM = 4;
    constexpr int64_t ACTION_DIM = 2;
    constexpr int EPISODES = 100;
    constexpr int TARGET_UPDATE_FREQ = 10;

    DummyEnv env;
    DQNAgent agent(STATE_DIM, ACTION_DIM, 10000);

    for (int episode = 1; episode <= EPISODES; ++episode) {
        torch::Tensor state = env.reset();
        float total_reward = 0.0f;
        bool done = false;

        while (!done) {
            int64_t action = agent.select_action(state);
            auto [next_state, reward, is_done] = env.step(action);

            agent.remember(state, action, reward, next_state, is_done);
            agent.train_step();

            state = next_state;
            total_reward += reward;
            done = is_done;
        }

        if (episode % TARGET_UPDATE_FREQ == 0) {
            agent.update_target_network();
        }

        std::cout << "Episode: " << episode << " | Total Reward: " << total_reward << std::endl;
    }

    return 0;
}