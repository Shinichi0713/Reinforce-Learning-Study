%%writefile /content/rl_cartpole.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <fstream>
#include <algorithm>

// ===================== 1. Tensor（行列） =====================
struct Tensor {
    std::vector<float> data;
    int rows, cols;
    Tensor(int r = 0, int c = 0) : rows(r), cols(c), data(r * c, 0.0f) {}
    float& operator()(int i, int j) { return data[i * cols + j]; }
    float operator()(int i, int j) const { return data[i * cols + j]; }
    int size() const { return rows * cols; }
};

// ===================== 2. 線形層（Linear） =====================
struct Linear {
    Tensor weight, bias, grad_w, grad_b;
    Tensor last_input;  // 逆伝播用に保存
    
    Linear(int in_feat, int out_feat) 
        : weight(out_feat, in_feat), bias(out_feat, 1),
          grad_w(out_feat, in_feat), grad_b(out_feat, 1) {
        // Xavier初期化
        std::mt19937 gen(123);
        float scale = std::sqrt(2.0f / (in_feat + out_feat));
        std::normal_distribution<float> dist(0, scale);
        for (auto& v : weight.data) v = dist(gen);
        for (auto& v : bias.data) v = 0.0f;
    }
    
    // 順伝播: y = xW^T + b
    Tensor forward(const Tensor& x) {
        last_input = x;
        Tensor out(x.rows, weight.rows);
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < weight.rows; j++) {
                float sum = bias(j, 0);
                for (int k = 0; k < x.cols; k++)
                    sum += x(i, k) * weight(j, k);
                out(i, j) = sum;
            }
        }
        return out;
    }
    
    // 勾配をゼロにリセット
    void zero_grad() {
        std::fill(grad_w.data.begin(), grad_w.data.end(), 0.0f);
        std::fill(grad_b.data.begin(), grad_b.data.end(), 0.0f);
    }
};

// ===================== 3. 活性化関数 =====================
void relu(Tensor& out, const Tensor& in) {
    for (int i = 0; i < in.size(); i++)
        out.data[i] = std::max(0.0f, in.data[i]);
}

void relu_backward(Tensor& grad_in, const Tensor& in, const Tensor& grad_out) {
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            grad_in(i, j) = (in(i, j) > 0) ? grad_out(i, j) : 0.0f;
}

// Softmax + log_softmax
void softmax(Tensor& probs, const Tensor& logits) {
    float max_val = *std::max_element(logits.data.begin(), logits.data.end());
    float sum = 0.0f;
    for (int i = 0; i < logits.cols; i++) {
        probs(0, i) = std::exp(logits(0, i) - max_val);
        sum += probs(0, i);
    }
    for (int i = 0; i < logits.cols; i++) probs(0, i) /= sum;
}

// ===================== 4. 逆伝播（線形層） =====================
void linear_backward(Linear& layer, const Tensor& grad_out) {
    const Tensor& input = layer.last_input;
    for (int i = 0; i < grad_out.rows; i++)
        for (int j = 0; j < grad_out.cols; j++)
            layer.grad_b(j, 0) += grad_out(i, j);
    
    for (int i = 0; i < layer.weight.rows; i++)
        for (int j = 0; j < layer.weight.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < grad_out.rows; k++)
                sum += grad_out(k, i) * input(k, j);
            layer.grad_w(i, j) += sum;
        }
}

// ===================== 5. Adamオプティマイザ =====================
struct Adam {
    float lr, b1, b2, eps;
    int t;
    Tensor m_w, v_w, m_b, v_b;
    
    Adam(Linear& layer, float lr_ = 1e-3)
        : lr(lr_), b1(0.9f), b2(0.999f), eps(1e-8f), t(0),
          m_w(layer.weight.rows, layer.weight.cols),
          v_w(layer.weight.rows, layer.weight.cols),
          m_b(layer.bias.rows, layer.bias.cols),
          v_b(layer.bias.rows, layer.bias.cols) {}
    
    void step(Linear& layer) {
        t++;
        float lr_t = lr * std::sqrt(1.0f - std::pow(b2, t)) / (1.0f - std::pow(b1, t));
        for (int i = 0; i < layer.weight.size(); i++) {
            m_w.data[i] = b1 * m_w.data[i] + (1 - b1) * layer.grad_w.data[i];
            v_w.data[i] = b2 * v_w.data[i] + (1 - b2) * layer.grad_w.data[i] * layer.grad_w.data[i];
            layer.weight.data[i] -= lr_t * m_w.data[i] / (std::sqrt(v_w.data[i]) + eps);
        }
        for (int i = 0; i < layer.bias.size(); i++) {
            m_b.data[i] = b1 * m_b.data[i] + (1 - b1) * layer.grad_b.data[i];
            v_b.data[i] = b2 * v_b.data[i] + (1 - b2) * layer.grad_b.data[i] * layer.grad_b.data[i];
            layer.bias.data[i] -= lr_t * m_b.data[i] / (std::sqrt(v_b.data[i]) + eps);
        }
    }
};

// ===================== 6. Actor-Critic ネットワーク =====================
struct ActorCritic {
    Linear shared1, shared2;
    Linear actor_head, critic_head;
    
    // 中間キャッシュ（逆伝播用）
    Tensor h1, h1_relu, h2, h2_relu, logits, value;
    
    ActorCritic(int state_dim, int hidden_dim, int action_dim)
        : shared1(state_dim, hidden_dim), shared2(hidden_dim, hidden_dim),
          actor_head(hidden_dim, action_dim), critic_head(hidden_dim, 1),
          h1(1, hidden_dim), h1_relu(1, hidden_dim), h2(1, hidden_dim), 
          h2_relu(1, hidden_dim), logits(1, action_dim), value(1, 1) {}
    
    // 順伝播: 状態 -> (行動logits, 状態価値)
    void forward(const Tensor& state) {
        h1 = shared1.forward(state);
        relu(h1_relu, h1);
        
        h2 = shared2.forward(h1_relu);
        relu(h2_relu, h2);
        
        logits = actor_head.forward(h2_relu);
        value = critic_head.forward(h2_relu);
    }
    
    // 勾配をゼロにリセット
    void zero_grad() {
        shared1.zero_grad(); shared2.zero_grad();
        actor_head.zero_grad(); critic_head.zero_grad();
    }
    
    // 逆伝播: 各ヘッドの勾配から入力方向へ伝播
    void backward(const Tensor& grad_actor_logits, const Tensor& grad_critic) {
        // Criticヘッドの逆伝播
        linear_backward(critic_head, grad_critic);
        Tensor grad_h2_critic(1, h2_relu.cols);
        for (int i = 0; i < grad_h2_critic.cols; i++) {
            grad_h2_critic(0, i) = 0.0f;
            for (int j = 0; j < critic_head.weight.rows; j++)
                grad_h2_critic(0, i) += grad_critic(0, j) * critic_head.weight(j, i);
        }
        
        // Actorヘッドの逆伝播
        linear_backward(actor_head, grad_actor_logits);
        Tensor grad_h2_actor(1, h2_relu.cols);
        for (int i = 0; i < grad_h2_actor.cols; i++) {
            grad_h2_actor(0, i) = 0.0f;
            for (int j = 0; j < actor_head.weight.rows; j++)
                grad_h2_actor(0, i) += grad_actor_logits(0, j) * actor_head.weight(j, i);
        }
        
        // 共有層への勾配合算
        Tensor grad_h2(1, h2_relu.cols);
        for (int i = 0; i < grad_h2.cols; i++)
            grad_h2(0, i) = grad_h2_critic(0, i) + grad_h2_actor(0, i);
        
        // ReLU2の逆伝播
        Tensor grad_h2_pre(1, h2.cols);
        relu_backward(grad_h2_pre, h2, grad_h2);
        linear_backward(shared2, grad_h2_pre);
        
        // ReLU1の逆伝播
        Tensor grad_h1(1, h1.cols);
        for (int i = 0; i < grad_h1.cols; i++) {
            grad_h1(0, i) = 0.0f;
            for (int j = 0; j < shared2.weight.rows; j++)
                grad_h1(0, i) += grad_h2_pre(0, j) * shared2.weight(j, i);
        }
        Tensor grad_h1_pre(1, h1.cols);
        relu_backward(grad_h1_pre, h1, grad_h1);
        linear_backward(shared1, grad_h1_pre);
    }
    
    // パラメータ更新
    void update(Adam& opt1, Adam& opt2, Adam& opt3, Adam& opt4) {
        opt1.step(shared1); opt2.step(shared2);
        opt3.step(actor_head); opt4.step(critic_head);
    }
};

// ===================== 7. CartPole 環境（C++実装） =====================
class CartPoleEnv {
public:
    float x, x_dot, theta, theta_dot;
    const float g = 9.8f, mc = 1.0f, mp = 0.1f;
    const float mt = mc + mp, l = 0.5f, mpl = mp * l;
    const float force_mag = 10.0f, dt = 0.02f;
    int step_count;
    
    CartPoleEnv() { reset(); }
    
    void reset() {
        std::mt19937 gen(static_cast<unsigned>(step_count + 1));
        std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
        x = dist(gen); x_dot = dist(gen);
        theta = dist(gen); theta_dot = dist(gen);
        step_count = 0;
    }
    
    std::vector<float> get_state() const {
        return {x, x_dot, theta, theta_dot};
    }
    
    // 行動を受け取り、報酬と終了フラグを返す
    int step(int action, float& reward) {
        float force = (action == 1) ? force_mag : -force_mag;
        float ct = std::cos(theta), st = std::sin(theta);
        
        float temp = (force + mpl * theta_dot * theta_dot * st) / mt;
        float thetaacc = (g * st - ct * temp) / 
                         (l * (4.0f/3.0f - mp * ct * ct / mt));
        float xacc = temp - mpl * thetaacc * ct / mt;
        
        x += dt * x_dot;
        x_dot += dt * xacc;
        theta += dt * theta_dot;
        theta_dot += dt * thetaacc;
        
        step_count++;
        bool done = (x < -2.4f || x > 2.4f || theta < -0.2095f || theta > 0.2095f);
        reward = 1.0f;
        return done ? 1 : 0;
    }
};

// ===================== 8. 学習ループ =====================
int main() {
    const int STATE_DIM = 4, HIDDEN_DIM = 128, ACTION_DIM = 2;
    const float GAMMA = 0.99f;
    const int MAX_EPISODES = 1000;
    const int MAX_STEPS = 500;
    
    ActorCritic ac(STATE_DIM, HIDDEN_DIM, ACTION_DIM);
    Adam opt1(ac.shared1, 1e-3), opt2(ac.shared2, 1e-3);
    Adam opt3(ac.actor_head, 1e-3), opt4(ac.critic_head, 1e-3);
    
    CartPoleEnv env;
    std::mt19937 rng(42);
    
    std::ofstream log_file("training_log.csv");
    log_file << "episode,total_reward,steps\n";
    
    std::cout << "=== C++ Actor-Critic: CartPole Training ===" << std::endl;
    
    for (int episode = 0; episode < MAX_EPISODES; episode++) {
        env.reset();
        Tensor state(1, STATE_DIM);
        state.data = env.get_state();
        
        float total_reward = 0.0f;
        int steps = 0;
        
        for (int t = 0; t < MAX_STEPS; t++) {
            // 順伝播
            ac.forward(state);
            
            // 行動確率の計算
            Tensor probs(1, ACTION_DIM);
            softmax(probs, ac.logits);
            
            // 行動サンプリング
            std::discrete_distribution<int> dist(probs.data.begin(), probs.data.end());
            int action = dist(rng);
            float log_prob = std::log(std::max(probs(0, action), 1e-8f));
            
            // 環境との相互作用
            float reward = 0.0f;
            int done = env.step(action, reward);
            total_reward += reward;
            steps++;
            
            // 次状態の価値（終了時は0）
            float next_value = 0.0f;
            if (!done) {
                Tensor next_state(1, STATE_DIM);
                next_state.data = env.get_state();
                ac.forward(next_state);
                next_value = ac.value(0, 0);
            }
            
            // TD誤差（Advantage）
            float current_value = ac.value(0, 0);
            float advantage = reward + GAMMA * next_value - current_value;
            
            // ===== 勾配計算 =====
            ac.zero_grad();
            
            // Actorの勾配: -advantage * ∇log π(a|s)
            Tensor grad_actor(1, ACTION_DIM);
            for (int i = 0; i < ACTION_DIM; i++)
                grad_actor(0, i) = -advantage * probs(0, i);
            grad_actor(0, action) += advantage;  // 選択した行動の勾配を調整
            // 上記は softmax cross-entropy の簡易版: -A * (1 - π(a))
            // 正確には: grad = -A * (δ_aj - π(j)) なので:
            for (int i = 0; i < ACTION_DIM; i++)
                grad_actor(0, i) = -advantage * probs(0, i);
            grad_actor(0, action) = -advantage * (probs(0, action) - 1.0f);
            
            // Criticの勾配: 2 * advantage * ∇V(s) （MSEの勾配）
            Tensor grad_critic(1, 1);
            grad_critic(0, 0) = 2.0f * advantage;
            
            // 逆伝播と更新
            ac.backward(grad_actor, grad_critic);
            ac.update(opt1, opt2, opt3, opt4);
            
            if (done) break;
            
            // 状態を更新
            state.data = env.get_state();
        }
        
        log_file << episode << "," << total_reward << "," << steps << "\n";
        
        if ((episode + 1) % 100 == 0) {
            std::cout << "Episode " << (episode + 1) << "/" << MAX_EPISODES 
                      << " | Reward: " << total_reward 
                      << " | Steps: " << steps << std::endl;
        }
    }
    
    log_file.close();
    std::cout << "\n学習完了。training_log.csv にログを保存しました。" << std::endl;
    
    // ===== 学習済みモデルの推論テスト =====
    std::cout << "\n=== 推論テスト（貪欲法） ===" << std::endl;
    env.reset();
    Tensor state(1, STATE_DIM);
    state.data = env.get_state();
    float test_reward = 0.0f;
    
    for (int t = 0; t < MAX_STEPS; t++) {
        ac.forward(state);
        Tensor probs(1, ACTION_DIM);
        softmax(probs, ac.logits);
        int action = (probs(0, 1) > probs(0, 0)) ? 1 : 0;  // 貪欲
        
        float reward = 0.0f;
        int done = env.step(action, reward);
        test_reward += reward;
        
        if (done) break;
        state.data = env.get_state();
    }
    std::cout << "テスト報酬: " << test_reward << " ステップ" << std::endl;
    
    return 0;
}