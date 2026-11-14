#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// ============================================================
// 強化学習：エントロピー正則化がパラメータ分布に与える影響のC++実装
// ============================================================
// ・PyTorch版の実の実験を純粋なC++で再現（外部ライブラリ不要）
// ・2行動バンディット環境 + 2層MLP方策ネットワーク
// ・エントロピー項なし(c=0.0) と あり(c=0.3) の2パターンを学習
// ・学習過程と最終パラメータをCSV出力（後でPython等で可視化能）
// ============================================================

const int INPUT_DIM = 1;
const int HIDDEN_DIM = 32;
const int OUTPUT_DIM = 2;
const int NUM_EPISODES = 500;
const double LR = 0.02;
const int SEED = 42;

// 乱数生成器
std::mt19937 global_gen(SEED);

// ------------------------------------------------------------------
// パラメータの統計量
// ------------------------------------------------------------------
struct ParamStats {
    double mean = 0.0;
    double std = 0.0;
    double min_val = 0.0;
    double max_val = 0.0;
    double l2 = 0.0;
};

ParamStats compute_stats(const std::vector<double>& v) {
    ParamStats s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    s.min_val = *std::min_element(v.begin(), v.end());
    s.max_val = *std::max_element(v.begin(), v.end());

    double sq_diff_sum = 0.0;
    double sq_sum = 0.0;
    for (double x : v) {
        sq_diff_sum += (x - s.mean) * (x - s.mean);
        sq_sum += x * x;
    }
    s.std = std::sqrt(sq_diff_sum / static_cast<double>(v.size()));
    s.l2 = std::sqrt(sq_sum);
    return s;
}

// ------------------------------------------------------------------
// 簡易バンディット環境
// ------------------------------------------------------------------
class SimpleBanditEnv {
public:
    // 行動0: 報酬1.0、行動1: 報酬0.6（ノイズ付き）
    double step(int action) {
        std::normal_distribution<double> noise(0.0, 0.1);
        double base_reward = (action == 0) ? 1.0 : 0.6;
        return base_reward + noise(global_gen);
    }
};

// ------------------------------------------------------------------
// 方策ネットワーク（MLP: 1 -> 32 -> 2）
// ------------------------------------------------------------------
class PolicyNetwork {
public:
    // パラメータ
    std::vector<double> fc1_w; // [HIDDEN_DIM]
    std::vector<double> fc1_b; // [HIDDEN_DIM]
    std::vector<double> fc2_w; // [OUTPUT_DIM * HIDDEN_DIM]
    std::vector<double> fc2_b; // [OUTPUT_DIM]

    // 勾配
    std::vector<double> g_fc1_w, g_fc1_b, g_fc2_w, g_fc2_b;

    // 中間値（forward保存）
    std::vector<double> hidden_pre; // tanh前
    std::vector<double> hidden;     // tanh後
    std::vector<double> logits;
    std::vector<double> probs;
    double input_val = 0.0;

    PolicyNetwork(int seed_offset = 0) {
        std::mt19937 init_gen(SEED + seed_offset);
        std::normal_distribution<double> dist(0.0, 0.1);

        fc1_w.resize(HIDDEN_DIM); for (auto& x : fc1_w) x = dist(init_gen);
        fc1_b.resize(HIDDEN_DIM); for (auto& x : fc1_b) x = dist(init_gen);
        fc2_w.resize(OUTPUT_DIM * HIDDEN_DIM); for (auto& x : fc2_w) x = dist(init_gen);
        fc2_b.resize(OUTPUT_DIM); for (auto& x : fc2_b) x = dist(init_gen);

        g_fc1_w.resize(HIDDEN_DIM, 0.0);
        g_fc1_b.resize(HIDDEN_DIM, 0.0);
        g_fc2_w.resize(OUTPUT_DIM * HIDDEN_DIM, 0.0);
        g_fc2_b.resize(OUTPUT_DIM, 0.0);

        hidden_pre.resize(HIDDEN_DIM);
        hidden.resize(HIDDEN_DIM);
        logits.resize(OUTPUT_DIM);
        probs.resize(OUTPUT_DIM);
    }

    // forward: 状態x（スカラー）を入力
    void forward(double x) {
        input_val = x;
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            hidden_pre[j] = fc1_w[j] * x + fc1_b[j];
            hidden[j] = std::tanh(hidden_pre[j]);
        }
        for (int k = 0; k < OUTPUT_DIM; ++k) {
            logits[k] = 0.0;
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                logits[k] += fc2_w[k * HIDDEN_DIM + j] * hidden[j];
            }
            logits[k] += fc2_b[k];
        }
        // softmax（数値安定のためmax引く）
        double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (int k = 0; k < OUTPUT_DIM; ++k) {
            probs[k] = std::exp(logits[k] - max_logit);
            sum_exp += probs[k];
        }
        for (int k = 0; k < OUTPUT_DIM; ++k) {
            probs[k] /= sum_exp;
        }
    }

    // backward: loss = -log_prob * reward - c_entropy * entropy
    void backward(int action, double reward, double c_entropy) {
        // エントロピー計算
        double entropy = 0.0;
        for (int k = 0; k < OUTPUT_DIM; ++k) {
            if (probs[k] > 1e-12) {
                entropy -= probs[k] * std::log(probs[k]);
            }
        }

        // d_logits[k] = -reward * (indicator - probs[k]) - c_entropy * probs[k] * (-log(probs[k]) - entropy)
        std::vector<double> d_logits(OUTPUT_DIM, 0.0);
        for (int k = 0; k < OUTPUT_DIM; ++k) {
            double indicator = (k == action) ? 1.0 : 0.0;
            double d_loss_log_prob = -reward * (indicator - probs[k]);
            double d_loss_entropy = -c_entropy * probs[k] * (-std::log(probs[k]) - entropy);
            d_logits[k] = d_loss_log_prob + d_loss_entropy;
        }

        // fc2の勾配 + hiddenへの勾配
        std::fill(g_fc2_w.begin(), g_fc2_w.end(), 0.0);
        std::fill(g_fc2_b.begin(), g_fc2_b.end(), 0.0);
        std::vector<double> d_hidden(HIDDEN_DIM, 0.0);

        for (int k = 0; k < OUTPUT_DIM; ++k) {
            for (int j = 0; j < HIDDEN_DIM; ++j) {
                g_fc2_w[k * HIDDEN_DIM + j] += d_logits[k] * hidden[j];
                d_hidden[j] += d_logits[k] * fc2_w[k * HIDDEN_DIM + j];
            }
            g_fc2_b[k] += d_logits[k];
        }

        // tanhの微分: d_tanh = d_hidden * (1 - tanh^2)
        std::fill(g_fc1_w.begin(), g_fc1_w.end(), 0.0);
        std::fill(g_fc1_b.begin(), g_fc1_b.end(), 0.0);
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            double d_tanh = d_hidden[j] * (1.0 - hidden[j] * hidden[j]);
            g_fc1_w[j] += d_tanh * input_val;
            g_fc1_b[j] += d_tanh;
        }
    }

    // SGD更新
    void sgd_update(double lr) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            fc1_w[j] -= lr * g_fc1_w[j];
            fc1_b[j] -= lr * g_fc1_b[j];
        }
        for (int k = 0; k < OUTPUT_DIM * HIDDEN_DIM; ++k) {
            fc2_w[k] -= lr * g_fc2_w[k];
        }
        for (int k = 0; k < OUTPUT_DIM; ++k) {
            fc2_b[k] -= lr * g_fc2_b[k];
        }
    }

    // 全パラメータをフラットに連結（ヒストグラム用）
    std::vector<double> get_all_params() const {
        std::vector<double> all;
        all.reserve(fc1_w.size() + fc1_b.size() + fc2_w.size() + fc2_b.size());
        all.insert(all.end(), fc1_w.begin(), fc1_w.end());
        all.insert(all.end(), fc1_b.begin(), fc1_b.end());
        all.insert(all.end(), fc2_w.begin(), fc2_w.end());
        all.insert(all.end(), fc2_b.begin(), fc2_b.end());
        return all;
    }
};

// ------------------------------------------------------------------
// 行動サンプリング
// ------------------------------------------------------------------
int sample_action(const std::vector<double>& probs) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(global_gen);
    double cum = 0.0;
    for (size_t k = 0; k < probs.size(); ++k) {
        cum += probs[k];
        if (r < cum) return static_cast<int>(k);
    }
    return static_cast<int>(probs.size()) - 1;
}

// ------------------------------------------------------------------
// CSV出力ヘルパー
// ------------------------------------------------------------------
void write_csv_header(std::ofstream& f) {
    f << "episode,probs_0,probs_1,entropy,reward,loss,"
      << "fc1_w_mean,fc1_w_std,fc1_w_min,fc1_w_max,fc1_w_l2,"
      << "fc1_b_mean,fc1_b_std,fc1_b_min,fc1_b_max,fc1_b_l2,"
      << "fc2_w_mean,fc2_w_std,fc2_w_min,fc2_w_max,fc2_w_l2,"
      << "fc2_b_mean,fc2_b_std,fc2_b_min,fc2_b_max,fc2_b_l2\n";
}

void write_csv_row(std::ofstream& f, int episode,
                   const std::vector<double>& probs,
                   double entropy, double reward, double loss,
                   const ParamStats& s_fc1w, const ParamStats& s_fc1b,
                   const ParamStats& s_fc2w, const ParamStats& s_fc2b) {
    f << episode << ","
      << std::fixed << std::setprecision(6)
      << probs[0] << "," << probs[1] << "," << entropy << "," << reward << "," << loss << ","
      << s_fc1w.mean << "," << s_fc1w.std << "," << s_fc1w.min_val << "," << s_fc1w.max_val << "," << s_fc1w.l2 << ","
      << s_fc1b.mean << "," << s_fc1b.std << "," << s_fc1b.min_val << "," << s_fc1b.max_val << "," << s_fc1b.l2 << ","
      << s_fc2w.mean << "," << s_fc2w.std << "," << s_fc2w.min_val << "," << s_fc2w.max_val << "," << s_fc2w.l2 << ","
      << s_fc2b.mean << "," << s_fc2b.std << "," << s_fc2b.min_val << "," << s_fc2b.max_val << "," << s_fc2b.l2 << "\n";
}

void write_final_params(const std::string& filename,
                        const std::vector<double>& fc1_w,
                        const std::vector<double>& fc1_b,
                        const std::vector<double>& fc2_w,
                        const std::vector<double>& fc2_b) {
    std::ofstream f(filename);
    f << "param_type,index,value\n";
    for (size_t i = 0; i < fc1_w.size(); ++i) f << "fc1.weight," << i << "," << fc1_w[i] << "\n";
    for (size_t i = 0; i < fc1_b.size(); ++i) f << "fc1.bias," << i << "," << fc1_b[i] << "\n";
    for (size_t i = 0; i < fc2_w.size(); ++i) f << "fc2.weight," << i << "," << fc2_w[i] << "\n";
    for (size_t i = 0; i < fc2_b.size(); ++i) f << "fc2.bias," << i << "," << fc2_b[i] << "\n";
}

// ------------------------------------------------------------------
// 学習ルプ
// ------------------------------------------------------------------
struct TrainResult {
    PolicyNetwork model;
    std::string csv_path;
    std::string final_params_path;
};

TrainResult train_reinforce(double c_entropy, const std::string& label,
                              const std::string& csv_filename,
                              const std::string& final_params_filename,
                              int seed_offset) {
    PolicyNetwork model(seed_offset);
    SimpleBanditEnv env;

    std::ofstream csv_file(csv_filename);
    write_csv_header(csv_file);

    std::cout << "\n=== " << label << " (c_entropy=" << c_entropy << ") ===" << std::endl;
    std::cout << std::setw(6) << "Ep" << std::setw(12) << "probs_0" << std::setw(12) << "probs_1"
              << std::setw(12) << "entropy" << std::setw(12) << "reward" << std::setw(14) << "fc2_w_std" << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    for (int ep = 0; ep < NUM_EPISODES; ++ep) {
        double state = 1.0; // ダミー入力
        model.forward(state);

        int action = sample_action(model.probs);
        double reward = env.step(action);

        // エントロピー
        double entropy = 0.0;
        for (int k = 0; k < OUTPUT_DIM; ++k) {
            if (model.probs[k] > 1e-12) {
                entropy -= model.probs[k] * std::log(model.probs[k]);
            }
        }

        // loss = -log_prob * reward - c_entropy * entropy
        double log_prob = std::log(model.probs[action] + 1e-10);
        double loss = -log_prob * reward - c_entropy * entropy;

        // backward & update
        model.backward(action, reward, c_entropy);
        model.sgd_update(LR);

        // 統計量計算
        ParamStats s_fc1w = compute_stats(model.fc1_w);
        ParamStats s_fc1b = compute_stats(model.fc1_b);
        ParamStats s_fc2w = compute_stats(model.fc2_w);
        ParamStats s_fc2b = compute_stats(model.fc2_b);

        // CSV書き込み（全エピソード）
        write_csv_row(csv_file, ep, model.probs, entropy, reward, loss,
                      s_fc1w, s_fc1b, s_fc2w, s_fc2b);

        // コンソール出力（10エピソードごと）
        if (ep % 10 == 0 || ep == NUM_EPISODES - 1) {
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(6) << ep
                      << std::setw(12) << model.probs[0]
                      << std::setw(12) << model.probs[1]
                      << std::setw(12) << entropy
                      << std::setw(12) << reward
                      << std::setw(14) << s_fc2w.std << std::endl;
        }
    }

    csv_file.close();

    // 最終パラメータ出力
    write_final_params(final_params_filename, model.fc1_w, model.fc1_b, model.fc2_w, model.fc2_b);

    std::cout << "CSV保存: " << csv_filename << std::endl;
    std::cout << "最終パラメータ保存: " << final_params_filename << std::endl;

    return {model, csv_filename, final_params_filename};
}

// ------------------------------------------------------------------
// メイン
// ------------------------------------------------------------------
int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "  強化学習: エントロピー正則化の効果検証（C++版）" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "設定:" << std::endl;
    std::cout << "  行動0（右）の報酬: 1.0" << std::endl;
    std::cout << "  行動1（左）の報酬: 0.6" << std::endl;
    std::cout << "  学習率: " << LR << std::endl;
    std::cout << "  エピソード数: " << NUM_EPISODES << std::endl;
    std::cout << "  ネットワーク: " << INPUT_DIM << " -> " << HIDDEN_DIM << " -> " << OUTPUT_DIM << std::endl;
    std::cout << "============================================================" << std::endl;

    // ケース1: エントロピー項なし
    TrainResult res_no = train_reinforce(
        0.0,
        "エントロピー項なし",
        "results_no_ent.csv",
        "params_final_no_ent.csv",
        0
    );

    // ケース2: エントロピー項あり
    TrainResult res_with = train_reinforce(
        0.3,
        "エントロピー項あり",
        "results_with_ent.csv",
        "params_final_with_ent.csv",
        1
    );

    // 最終比較
    std::cout << "\n============================================================" << std::endl;
    std::cout << "【最終パラメータ統計比較】" << std::endl;
    std::cout << "============================================================" << std::endl;

    auto print_stats = [](const std::string& name, const ParamStats& s_no, const ParamStats& s_with) {
        std::cout << "--- " << name << " ---" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  なし: mean=" << s_no.mean << " std=" << s_no.std 
                  << " min=" << s_no.min_val << " max=" << s_no.max_val << " l2=" << s_no.l2 << std::endl;
        std::cout << "  あり: mean=" << s_with.mean << " std=" << s_with.std 
                  << " min=" << s_with.min_val << " max=" << s_with.max_val << " l2=" << s_with.l2 << std::endl;
    };

    ParamStats s_no_fc1w = compute_stats(res_no.model.fc1_w);
    ParamStats s_no_fc1b = compute_stats(res_no.model.fc1_b);
    ParamStats s_no_fc2w = compute_stats(res_no.model.fc2_w);
    ParamStats s_no_fc2b = compute_stats(res_no.model.fc2_b);

    ParamStats s_with_fc1w = compute_stats(res_with.model.fc1_w);
    ParamStats s_with_fc1b = compute_stats(res_with.model.fc1_b);
    ParamStats s_with_fc2w = compute_stats(res_with.model.fc2_w);
    ParamStats s_with_fc2b = compute_stats(res_with.model.fc2_b);

    print_stats("fc1.weight", s_no_fc1w, s_with_fc1w);
    print_stats("fc1.bias", s_no_fc1b, s_with_fc1b);
    print_stats("fc2.weight", s_no_fc2w, s_with_fc2w);
    print_stats("fc2.bias", s_no_fc2b, s_with_fc2b);

    std::cout << "\n【結果の解説】" << std::endl;
    std::cout << "  ・fc2.weight / fc2.bias の std/l2 が「あり」の方が小さい場合、" << std::endl;
    std::cout << "    エントロピー正則化が「重みの極端な成長」を抑制できている証拠です。" << std::endl;
    std::cout << "  ・results_*.csv をPython(matplotlib)/Excelで読み込み、" << std::endl;
    std::cout << "    行動確率・エントロピー・L2ノルムの推移をグラフ化できます。" << std::endl;
    std::cout << "  ・params_final_*.csv を読み込んでヒストグラムを描くことで、" << std::endl;
    std::cout << "    パラメータ分布の形を比較できます。" << std::endl;
    std::cout << "============================================================" << std::endl;

    return 0;
}
