%%writefile cartpole_control.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <onnxruntime_cxx_api.h>

// --- 1. C++ 側で実装した CartPole 物理環境 ---
class CartPoleEnv {
public:
    double x = 0.0;          // カートの位置 [m]
    double x_dot = 0.0;      // カートの速度 [m/s]
    double theta = 0.05;     // ポールの角度 [rad] (初期状態: 微小な傾き)
    double theta_dot = 0.0;  // ポールの角速度 [rad/s]

    const double gravity = 9.8;
    const double masscart = 1.0;
    const double masspole = 0.1;
    const double total_mass = masscart + masspole;
    const double length = 0.5; // ポールの半分の長さ
    const double polemass_length = masspole * length;
    const double force_mag = 10.0;
    const double tau = 0.02;   // 1ステップの時間幅 [s]

    // 行動に応じたオイラー法でのステップ更新 (0: 左へ押す, 1: 右へ押す)
    bool step(int action) {
        double force = (action == 1) ? force_mag : -force_mag;
        double costheta = std::cos(theta);
        double sintheta = std::sin(theta);

        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double thetaacc = (gravity * sintheta - costheta * temp) / 
                           (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        // 状態の更新
        x += tau * x_dot;
        x_dot += tau * xacc;
        theta += tau * theta_dot;
        theta_dot += tau * thetaacc;

        // 終了条件チェック (倒れるか画面外に出たら終了)
        bool failed = (x < -2.4 || x > 2.4 || theta < -0.2095 || theta > 0.2095);
        return !failed;
    }

    std::vector<float> get_state() const {
        return { static_cast<float>(x), static_cast<float>(x_dot), 
                 static_cast<float>(theta), static_cast<float>(theta_dot) };
    }
};

// --- 2. メイン制御ループ (ONNX Runtime 推論) ---
int main() {
    // ONNX Runtime の設定
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CartPoleInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // モデル読み込み
    const char* model_path = "cartpole_dqn.onnx";
    Ort::Session session(env, model_path, session_options);

    CartPoleEnv cartpole;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* input_names[] = {"state"};
    const char* output_names[] = {"q_values"};
    std::vector<int64_t> input_shape = {1, 4};

    std::cout << "--- C++ 側での自律制御ループを開始 ---" << std::endl;
    int steps = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (steps < 500) {
        // 現在の状態を取得
        std::vector<float> state = cartpole.get_state();

        // C++ テンソルの構築
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, state.data(), state.size(), input_shape.data(), input_shape.size());

        // ONNX モデル推論の呼び出し
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // 推論結果（Q値）の解釈
        float* q_values = output_tensors[0].GetTensorMutableData<float>();
        int action = (q_values[1] > q_values[0]) ? 1 : 0; // Argmax (決定論的選択)

        // C++ 物理シミュレータを進める
        bool alive = cartpole.step(action);
        steps++;

        if (!alive) {
            std::cout << "制御失敗: ステップ " << steps << " で倒れました。" << std::endl;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (steps >= 500) {
        std::cout << "🎉 制御成功！最大500ステップ維持を達成しました。" << std::endl;
    }
    std::cout << "総ステップ数: " << steps << std::endl;
    std::cout << "500ステップ推論・シミュレーションの総計算時間: " 
              << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "1ステップあたりの平均処理時間: " 
              << (duration.count() / static_cast<double>(steps)) << " μs" << std::endl;

    return 0;
}