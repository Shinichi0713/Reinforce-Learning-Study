#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>
#include <numeric>

struct Tensor {
    std::vector<float> data;
    int rows, cols;
    
    Tensor(int r = 0, int c = 0) : rows(r), cols(c), data(r * c, 0.0f) {}
    
    float& operator()(int i, int j) { return data[i * cols + j]; }
    float operator()(int i, int j) const { return data[i * cols + j]; }
    
    int size() const { return rows * cols; }
};

// ReLU: max(0, x)
void relu(Tensor& out, const Tensor& in) {
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            out(i, j) = std::max(0.0f, in(i, j));
}

// ReLUの微分
void relu_backward(Tensor& grad_in, const Tensor& in, const Tensor& grad_out) {
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            grad_in(i, j) = (in(i, j) > 0) ? grad_out(i, j) : 0.0f;
}

// Tanh
void tanh(Tensor& out, const Tensor& in) {
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            out(i, j) = std::tanh(in(i, j));
}

// Tanhの微分: 1 - tanh(x)^2
void tanh_backward(Tensor& grad_in, const Tensor& out, const Tensor& grad_out) {
    for (int i = 0; i < out.rows; i++)
        for (int j = 0; j < out.cols; j++) {
            float t = out(i, j);
            grad_in(i, j) = (1.0f - t * t) * grad_out(i, j);
        }
}

struct Linear {
    Tensor weight;  // [out_features, in_features]
    Tensor bias;    // [out_features, 1]
    
    // 勾配（学習用）
    Tensor grad_weight;
    Tensor grad_bias;
    
    // 順伝播時の入力を保存（逆伝播用）
    Tensor last_input;
    Tensor last_pre_activation;  // 活性化前の値
    
    Linear(int in_feat, int out_feat) 
        : weight(out_feat, in_feat), bias(out_feat, 1),
          grad_weight(out_feat, in_feat), grad_bias(out_feat, 1) {
        // Xavier初期化
        std::mt19937 gen(42);
        float scale = std::sqrt(2.0f / (in_feat + out_feat));
        std::normal_distribution<float> dist(0, scale);
        for (auto& v : weight.data) v = dist(gen);
        for (auto& v : bias.data) v = 0.0f;
    }
    
    // 順伝播: out = x * W^T + b
    void forward(Tensor& out, const Tensor& x) {
        last_input = x;
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < weight.rows; j++) {  // weight.rows = out_features
                float sum = bias(j, 0);
                for (int k = 0; k < x.cols; k++)      // x.cols = in_features
                    sum += x(i, k) * weight(j, k);
                out(i, j) = sum;
            }
        }
        last_pre_activation = out;
    }
};

struct MLP {
    Linear fc1, fc2, fc3;
    Tensor h1, h2, h3;        // 順伝播の中間出力
    Tensor h1_pre, h2_pre;  // 活性化前
    
    MLP(int input_dim, int hidden_dim, int output_dim)
        : fc1(input_dim, hidden_dim),
          fc2(hidden_dim, hidden_dim),
          fc3(hidden_dim, output_dim),
          h1(1, hidden_dim), h2(1, hidden_dim), h3(1, output_dim),
          h1_pre(1, hidden_dim), h2_pre(1, hidden_dim) {}
    
    // 順伝播: input [1, input_dim] -> output [1, output_dim]
    Tensor forward(const Tensor& input) {
        fc1.forward(h1_pre, input);
        relu(h1, h1_pre);
        
        fc2.forward(h2_pre, h1);
        relu(h2, h2_pre);
        
        fc3.forward(h3, h2);
        tanh(h3, h3);  // Actor: 出力を -1 ~ +1 に制限
        
        return h3;
    }
};

float mse_loss(Tensor& grad_pred, const Tensor& pred, const Tensor& target) {
    float loss = 0.0f;
    for (int i = 0; i < pred.size(); i++) {
        float diff = pred.data[i] - target.data[i];
        grad_pred.data[i] = 2.0f * diff / pred.size();
        loss += diff * diff;
    }
    return loss / pred.size();
}


void linear_backward(Linear& layer, const Tensor& grad_out, const Tensor& input) {
    // biasの勾配
    for (int i = 0; i < grad_out.rows; i++)
        for (int j = 0; j < grad_out.cols; j++)
            layer.grad_bias(j, 0) += grad_out(i, j);
    
    // weightの勾配: grad_W = grad_out^T * input
    for (int i = 0; i < layer.weight.rows; i++) {
        for (int j = 0; j < layer.weight.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < grad_out.rows; k++)
                sum += grad_out(k, i) * input(k, j);
            layer.grad_weight(i, j) += sum;
        }
    }
}

struct Adam {
    float lr, beta1, beta2, eps;
    int t;
    
    Tensor m_w, v_w, m_b, v_b;
    
    Adam(Linear& layer, float learning_rate = 1e-3)
        : lr(learning_rate), beta1(0.9f), beta2(0.999f), eps(1e-8f), t(0),
          m_w(layer.weight.rows, layer.weight.cols),
          v_w(layer.weight.rows, layer.weight.cols),
          m_b(layer.bias.rows, layer.bias.cols),
          v_b(layer.bias.rows, layer.bias.cols) {}
    
    void step(Linear& layer) {
        t++;
        float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));
        
        for (int i = 0; i < layer.weight.size(); i++) {
            m_w.data[i] = beta1 * m_w.data[i] + (1 - beta1) * layer.grad_weight.data[i];
            v_w.data[i] = beta2 * v_w.data[i] + (1 - beta2) * layer.grad_weight.data[i] * layer.grad_weight.data[i];
            layer.weight.data[i] -= lr_t * m_w.data[i] / (std::sqrt(v_w.data[i]) + eps);
        }
        
        for (int i = 0; i < layer.bias.size(); i++) {
            m_b.data[i] = beta1 * m_b.data[i] + (1 - beta1) * layer.grad_bias.data[i];
            v_b.data[i] = beta2 * v_b.data[i] + (1 - beta2) * layer.grad_bias.data[i] * layer.grad_bias.data[i];
            layer.bias.data[i] -= lr_t * m_b.data[i] / (std::sqrt(v_b.data[i]) + eps);
        }
        
        // 勾配をゼロにリセット
        std::fill(layer.grad_weight.data.begin(), layer.grad_weight.data.end(), 0.0f);
        std::fill(layer.grad_bias.data.begin(), layer.grad_bias.data.end(), 0.0f);
    }
};


%%writefile /content/rl_model.cpp

#include <vector>
#include <iostream>
#include <cmath>
#include <random>
#include <cassert>
#include <numeric>
#include <algorithm>

// ===================== Step 1: Tensor =====================
struct Tensor {
    std::vector<float> data;
    int rows, cols;
    
    Tensor(int r = 0, int c = 0) : rows(r), cols(c), data(r * c, 0.0f) {}
    
    float& operator()(int i, int j) { return data[i * cols + j]; }
    float operator()(int i, int j) const { return data[i * cols + j]; }
    int size() const { return rows * cols; }
};

// ===================== Step 2: Activations =====================
void relu(Tensor& out, const Tensor& in) {
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            out(i, j) = std::max(0.0f, in(i, j));
}

void relu_backward(Tensor& grad_in, const Tensor& in, const Tensor& grad_out) {
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            grad_in(i, j) = (in(i, j) > 0) ? grad_out(i, j) : 0.0f;
}

void tanh_forward(Tensor& out, const Tensor& in) {
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            out(i, j) = std::tanh(in(i, j));
}

void tanh_backward(Tensor& grad_in, const Tensor& out, const Tensor& grad_out) {
    for (int i = 0; i < out.rows; i++)
        for (int j = 0; j < out.cols; j++) {
            float t = out(i, j);
            grad_in(i, j) = (1.0f - t * t) * grad_out(i, j);
        }
}

// ===================== Step 3: Linear Layer =====================
struct Linear {
    Tensor weight, bias;
    Tensor grad_weight, grad_bias;
    
    Linear(int in_feat, int out_feat) 
        : weight(out_feat, in_feat), bias(out_feat, 1),
          grad_weight(out_feat, in_feat), grad_bias(out_feat, 1) {
        std::mt19937 gen(42);
        float scale = std::sqrt(2.0f / (in_feat + out_feat));
        std::normal_distribution<float> dist(0, scale);
        for (auto& v : weight.data) v = dist(gen);
        for (auto& v : bias.data) v = 0.0f;
    }
    
    void forward(Tensor& out, const Tensor& x) {
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < weight.rows; j++) {
                float sum = bias(j, 0);
                for (int k = 0; k < x.cols; k++)
                    sum += x(i, k) * weight(j, k);
                out(i, j) = sum;
            }
        }
    }
};

// ===================== Step 6: Backprop =====================
void linear_backward(Linear& layer, const Tensor& grad_out, const Tensor& input) {
    for (int i = 0; i < grad_out.rows; i++)
        for (int j = 0; j < grad_out.cols; j++)
            layer.grad_bias(j, 0) += grad_out(i, j);
    
    for (int i = 0; i < layer.weight.rows; i++)
        for (int j = 0; j < layer.weight.cols; j++) {
            float sum = 0.0f;
            for (int k = 0; k < grad_out.rows; k++)
                sum += grad_out(k, i) * input(k, j);
            layer.grad_weight(i, j) += sum;
        }
}

// ===================== Step 7: Adam Optimizer =====================
struct Adam {
    float lr, beta1, beta2, eps;
    int t;
    Tensor m_w, v_w, m_b, v_b;
    
    Adam(Linear& layer, float learning_rate = 1e-3)
        : lr(learning_rate), beta1(0.9f), beta2(0.999f), eps(1e-8f), t(0),
          m_w(layer.weight.rows, layer.weight.cols),
          v_w(layer.weight.rows, layer.weight.cols),
          m_b(layer.bias.rows, layer.bias.cols),
          v_b(layer.bias.rows, layer.bias.cols) {}
    
    void step(Linear& layer) {
        t++;
        float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, t)) / (1.0f - std::pow(beta1, t));
        
        for (int i = 0; i < layer.weight.size(); i++) {
            m_w.data[i] = beta1 * m_w.data[i] + (1 - beta1) * layer.grad_weight.data[i];
            v_w.data[i] = beta2 * v_w.data[i] + (1 - beta2) * layer.grad_weight.data[i] * layer.grad_weight.data[i];
            layer.weight.data[i] -= lr_t * m_w.data[i] / (std::sqrt(v_w.data[i]) + eps);
        }
        
        for (int i = 0; i < layer.bias.size(); i++) {
            m_b.data[i] = beta1 * m_b.data[i] + (1 - beta1) * layer.grad_bias.data[i];
            v_b.data[i] = beta2 * v_b.data[i] + (1 - beta2) * layer.grad_bias.data[i] * layer.grad_bias.data[i];
            layer.bias.data[i] -= lr_t * m_b.data[i] / (std::sqrt(v_b.data[i]) + eps);
        }
        
        std::fill(layer.grad_weight.data.begin(), layer.grad_weight.data.end(), 0.0f);
        std::fill(layer.grad_bias.data.begin(), layer.grad_bias.data.end(), 0.0f);
    }
};

// ===================== Step 4 & 8: MLP Actor =====================
struct Actor {
    Linear fc1, fc2, fc3;
    Tensor h1, h2, out;
    Tensor h1_pre, h2_pre;
    
    Actor(int input_dim, int hidden_dim, int output_dim)
        : fc1(input_dim, hidden_dim), fc2(hidden_dim, hidden_dim), fc3(hidden_dim, output_dim),
          h1(1, hidden_dim), h2(1, hidden_dim), out(1, output_dim),
          h1_pre(1, hidden_dim), h2_pre(1, hidden_dim) {}
    
    // 順伝播
    Tensor forward(const Tensor& input) {
        fc1.forward(h1_pre, input);
        relu(h1, h1_pre);
        
        fc2.forward(h2_pre, h1);
        relu(h2, h2_pre);
        
        fc3.forward(out, h2);
        tanh_forward(out, out);
        
        return out;
    }
    
    // 逆伝播（Actorの場合は -grad_Q を使うが、ここでは教師あり学習風にMSEでデモ）
    void backward(const Tensor& grad_out) {
        Tensor grad_h2(1, h2.cols), grad_h1(1, h1.cols);
        Tensor grad_h2_pre(1, h2_pre.cols), grad_h1_pre(1, h1_pre.cols);
        
        tanh_backward(grad_h2_pre, out, grad_out);
        linear_backward(fc3, grad_h2_pre, h2);
        
        relu_backward(grad_h2, h2_pre, grad_h2_pre);
        linear_backward(fc2, grad_h2, h1);
        
        relu_backward(grad_h1, h1_pre, grad_h2);
        linear_backward(fc1, grad_h1, Tensor(1, fc1.weight.cols));  // 入力勾配は使わない
    }
};

// ===================== Step 5: Loss =====================
float mse_loss(Tensor& grad_pred, const Tensor& pred, const Tensor& target) {
    float loss = 0.0f;
    for (int i = 0; i < pred.size(); i++) {
        float diff = pred.data[i] - target.data[i];
        grad_pred.data[i] = 2.0f * diff / pred.size();
        loss += diff * diff;
    }
    return loss / pred.size();
}

// ===================== Main: 学習デモ =====================
int main() {
    // Pendulum-v1 相当: 観測3次元 -> 行動1次元
    Actor actor(3, 64, 1);
    Adam adam1(actor.fc1), adam2(actor.fc2), adam3(actor.fc3);
    
    // ダミーデータ: 観測 -> 目標行動（教師ありで学習させるデモ）
    std::vector<std::pair<std::vector<float>, float>> dataset = {
        {{1.0f, 0.0f, 0.1f},  0.5f},
        {{0.0f, 1.0f, -0.1f}, -0.3f},
        {{-1.0f, 0.0f, 0.2f}, 0.8f},
        {{0.0f, -1.0f, -0.2f}, -0.7f}
    };
    
    std::cout << "=== C++ Actor Model Training Demo ===" << std::endl;
    
    for (int epoch = 0; epoch < 500; epoch++) {
        float total_loss = 0.0f;
        
        for (auto& sample : dataset) {
            Tensor input(1, 3);
            input.data = sample.first;
            float target_val = sample.second;
            
            // 順伝播
            Tensor pred = actor.forward(input);
            
            // 損失計算
            Tensor target(1, 1);
            target(0, 0) = target_val;
            Tensor grad_pred(1, 1);
            float loss = mse_loss(grad_pred, pred, target);
            total_loss += loss;
            
            // 逆伝播
            actor.backward(grad_pred);
            
            // パラメータ更新
            adam1.step(actor.fc1);
            adam2.step(actor.fc2);
            adam3.step(actor.fc3);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << total_loss / dataset.size() << std::endl;
        }
    }
    
    // 推論テスト
    std::cout << "\n=== Inference Test ===" << std::endl;
    Tensor test_input(1, 3);
    test_input.data = {1.0f, 0.0f, 0.1f};
    Tensor result = actor.forward(test_input);
    std::cout << "Input: [1.0, 0.0, 0.1] -> Action: " << result(0, 0) << std::endl;
    
    std::cout << "\nモデル構築・学習・推論が純粋C++で完了しました。" << std::endl;
    
    return 0;
}