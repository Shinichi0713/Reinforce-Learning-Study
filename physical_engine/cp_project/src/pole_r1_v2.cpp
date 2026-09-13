%%writefile cartpole_control.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream> // CSV出力用
#include <onnxruntime_cxx_api.h>

class CartPoleEnv {
public:
    double x = 0.0;
    double x_dot = 0.0;
    double theta = 0.05;
    double theta_dot = 0.0;

    const double gravity = 9.8;
    const double masscart = 1.0;
    const double masspole = 0.1;
    const double total_mass = masscart + masspole;
    const double length = 0.5;
    const double polemass_length = masspole * length;
    const double force_mag = 10.0;
    const double tau = 0.02;

    bool step(int action) {
        double force = (action == 1) ? force_mag : -force_mag;
        double costheta = std::cos(theta);
        double sintheta = std::sin(theta);

        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double thetaacc = (gravity * sintheta - costheta * temp) / 
                           (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double xacc = temp - polemass_length * thetaacc * costheta / total_mass;

        x += tau * x_dot;
        x_dot += tau * xacc;
        theta += tau * theta_dot;
        theta_dot += tau * thetaacc;

        bool failed = (x < -2.4 || x > 2.4 || theta < -0.2095 || theta > 0.2095);
        return !failed;
    }

    std::vector<float> get_state() const {
        return { static_cast<float>(x), static_cast<float>(x_dot), 
                 static_cast<float>(theta), static_cast<float>(theta_dot) };
    }
};

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CartPoleInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    const char* model_path = "cartpole_dqn.onnx";
    Ort::Session session(env, model_path, session_options);

    CartPoleEnv cartpole;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    const char* input_names[] = {"state"};
    const char* output_names[] = {"q_values"};
    std::vector<int64_t> input_shape = {1, 4};

    // CSV ファイルを開く
    std::ofstream log_file("cartpole_log.csv");
    log_file << "step,x,theta,action\n";

    int steps = 0;
    while (steps < 500) {
        std::vector<float> state = cartpole.get_state();

        // ログ書き出し (ステップ数, カート位置, ポール角度)
        log_file << steps << "," << cartpole.x << "," << cartpole.theta << ",0\n";

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, state.data(), state.size(), input_shape.data(), input_shape.size());

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float* q_values = output_tensors[0].GetTensorMutableData<float>();
        int action = (q_values[1] > q_values[0]) ? 1 : 0;

        bool alive = cartpole.step(action);
        steps++;

        if (!alive) break;
    }
    log_file.close();

    std::cout << "ログ出力完了: " << steps << " ステップ分のデータを出力しました。" << std::endl;
    return 0;
}