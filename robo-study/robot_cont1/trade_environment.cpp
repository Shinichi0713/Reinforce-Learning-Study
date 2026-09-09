
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <iomanip>

// ============================================================
// 強化学習用 株価トレード環境 (C++)
// ============================================================

struct TradeEnvironment {
    // 市場データ
    std::vector<double> prices;      // 終値の時系列
    std::vector<double> highs;       // 高値
    std::vector<double> lows;        // 安値
    std::vector<double> volumes;     // 出来高

    // エージェントの状態
    int current_step = 0;            // 現在のタイムステップ
    double balance = 0.0;            // 現金残高
    int shares_held = 0;             // 保有株数
    double total_profit = 0.0;       // 累積利益
    double initial_balance = 0.0;    // 初期資金

    // 取引設定
    double transaction_fee_percent = 0.001; // 手数料 (0.1%)
    int window_size = 10;            // 観測ウィンドウサイズ

    // 履歴
    std::vector<double> portfolio_values; // 資産総額の履歴
    std::vector<int> actions_history;     // アクション履歴

    // 乱数生成
    std::mt19937 rng;

    // --------------------------------------------------------
    // コンストラクタ
    // --------------------------------------------------------
    TradeEnvironment(const std::vector<double>& price_data,
                     double init_balance = 100000.0,
                     double fee = 0.001,
                     int window = 10)
        : prices(price_data),
          balance(init_balance),
          initial_balance(init_balance),
          transaction_fee_percent(fee),
          window_size(window),
          rng(42)
    {
        // 高値・安値・出来高を価格から近似生成（実データがあれば置き換え）
        std::uniform_real_distribution<double> dist(-0.02, 0.02);
        std::uniform_real_distribution<double> vol_dist(1000, 10000);
        for (size_t i = 0; i < prices.size(); ++i) {
            highs.push_back(prices[i] * (1.0 + std::abs(dist(rng))));
            lows.push_back(prices[i] * (1.0 - std::abs(dist(rng))));
            volumes.push_back(vol_dist(rng));
        }
    }

    // --------------------------------------------------------
    // 環境を初期状態にリセット
    // --------------------------------------------------------
    std::vector<double> reset() {
        current_step = window_size;
        balance = initial_balance;
        shares_held = 0;
        total_profit = 0.0;
        portfolio_values.clear();
        actions_history.clear();
        portfolio_values.push_back(initial_balance);
        return get_observation();
    }

    // --------------------------------------------------------
    // 観測状態の取得
    // 正規化された特徴量ベクトルを返す
    // --------------------------------------------------------
    std::vector<double> get_observation() {
        std::vector<double> obs;
        obs.reserve(window_size * 5 + 2);

        // ウィンドウ内の価格データを正規化
        double base_price = prices[current_step];
        for (int i = current_step - window_size; i < current_step; ++i) {
            obs.push_back((prices[i] - base_price) / base_price);  // 終値
            obs.push_back((highs[i] - base_price) / base_price);   // 高値
            obs.push_back((lows[i] - base_price) / base_price);    // 安値
            // 出来高の対数正規化
            obs.push_back(std::log(volumes[i] + 1.0) / 10.0);
            // 日次リターン
            double ret = (prices[i+1] - prices[i]) / prices[i];
            obs.push_back(ret * 10.0);
        }

        // ポートフォリオ状態
        double portfolio_value = balance + shares_held * prices[current_step];
        obs.push_back(balance / initial_balance);
        obs.push_back(static_cast<double>(shares_held) * prices[current_step] / initial_balance);

        return obs;
    }

    // --------------------------------------------------------
    // 1ステップ進める
    // action: 0=HOLD(保有), 1=BUY(買い), 2=SELL(売り)
    // 返り値: {次の観測状態, 報酬, 終了フラグ, 情報}
    // --------------------------------------------------------
    struct StepResult {
        std::vector<double> observation;
        double reward;
        bool done;
        std::string info;
    };

    StepResult step(int action) {
        double current_price = prices[current_step];
        double prev_portfolio = balance + shares_held * current_price;

        std::string trade_info = "NONE";

        // アクション実行
        if (action == 1) { // BUY
            if (balance > current_price) {
                int max_shares = static_cast<int>(balance / (current_price * (1 + transaction_fee_percent)));
                if (max_shares > 0) {
                    double cost = max_shares * current_price * (1 + transaction_fee_percent);
                    balance -= cost;
                    shares_held += max_shares;
                    trade_info = "BUY " + std::to_string(max_shares);
                }
            }
        } else if (action == 2) { // SELL
            if (shares_held > 0) {
                double revenue = shares_held * current_price * (1 - transaction_fee_percent);
                balance += revenue;
                trade_info = "SELL " + std::to_string(shares_held);
                shares_held = 0;
            }
        }

        // 次のステップへ
        current_step++;
        double next_price = prices[current_step];
        double new_portfolio = balance + shares_held * next_price;
        portfolio_values.push_back(new_portfolio);
        actions_history.push_back(action);

        // 報酬計算: ポートフォリオ価値の変化率
        double reward = (new_portfolio - prev_portfolio) / prev_portfolio;
        // スケーリング
        reward *= 100.0;

        // 終了判定
        bool done = (current_step >= static_cast<int>(prices.size()) - 1);
        if (done) {
            total_profit = new_portfolio - initial_balance;
        }

        std::string info = "Action:" + std::to_string(action) +
                           " Trade:" + trade_info +
                           " Price:" + std::to_string(current_price) +
                           " Portfolio:" + std::to_string(new_portfolio);

        return {get_observation(), reward, done, info};
    }

    // --------------------------------------------------------
    // ポートフォリオ価値の取得
    // --------------------------------------------------------
    double get_portfolio_value() const {
        return balance + shares_held * prices[current_step];
    }

    // --------------------------------------------------------
    // パフォーマンスサマリーの出力
    // --------------------------------------------------------
    void print_summary() const {
        double final_value = balance + shares_held * prices[current_step];
        double total_return = (final_value - initial_balance) / initial_balance * 100.0;
        double buy_hold_return = (prices.back() - prices[window_size]) / prices[window_size] * 100.0;

        std::cout << "\n========== トレード環境 サマリー ==========\n";
        std::cout << "初期資金:        " << std::fixed << std::setprecision(2) << initial_balance << "\n";
        std::cout << "最終資産価値:    " << final_value << "\n";
        std::cout << "総利益:          " << total_profit << "\n";
        std::cout << "総リターン:      " << total_return << "%\n";
        std::cout << "Buy & Hold返り:  " << buy_hold_return << "%\n";
        std::cout << "保有株数:        " << shares_held << "\n";
        std::cout << "現金残高:        " << balance << "\n";
        std::cout << "取引回数:        " << actions_history.size() << "\n";

        // シャープレシオ（簡易版）
        if (portfolio_values.size() > 1) {
            std::vector<double> returns;
            for (size_t i = 1; i < portfolio_values.size(); ++i) {
                returns.push_back((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]);
            }
            double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
            double sq_sum = 0.0;
            for (double r : returns) sq_sum += (r - mean) * (r - mean);
            double stddev = std::sqrt(sq_sum / returns.size());
            double sharpe = (stddev > 0) ? (mean / stddev) * std::sqrt(252.0) : 0.0; // 年率換算
            std::cout << "シャープレシオ:  " << sharpe << "\n";
        }
        std::cout << "==========================================\n";
    }

    // --------------------------------------------------------
    // 測空間の次元数
    // --------------------------------------------------------
    int observation_dim() const {
        return window_size * 5 + 2;
    }

    // --------------------------------------------------------
    // アクション空間の次元数
    // --------------------------------------------------------
    static int action_dim() {
        return 3; // HOLD, BUY, SELL
    }
};

// ============================================================
// ダミー株価データ生成（ランダムウォーク）
// ============================================================
std::vector<double> generate_dummy_prices(int n = 500, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 0.02);
    std::vector<double> prices;
    prices.reserve(n);
    double price = 100.0;
    for (int i = 0; i < n; ++i) {
        prices.push_back(price);
        price *= (1.0 + dist(gen));
        if (price < 10.0) price = 10.0;
    }
    return prices;
}

// ============================================================
// ランダムエージェントによるテスト
// ============================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "  強化学習用 株価トレード環境 (C++)\n";
    std::cout << "========================================\n\n";

    // ダミー株価データの生成
    auto prices = generate_dummy_prices(500);
    std::cout << "生成した株価データ点数: " << prices.size() << "\n";
    std::cout << "最初の価格: " << prices[0] << "\n";
    std::cout << "最後の価格: " << prices.back() << "\n\n";

    // 環境の構築
    TradeEnvironment env(prices, 100000.0, 0.001, 10);

    std::cout << "観測空間の次元数: " << env.observation_dim() << "\n";
    std::cout << "アクション空間の次元数: " << env.action_dim() << " (0=HOLD, 1=BUY, 2=SELL)\n\n";

    // 環境リセット
    auto obs = env.reset();
    std::cout << "初期観測ベクトル (先頭10要素):\n";
    for (size_t i = 0; i < std::min(obs.size(), size_t(10)); ++i) {
        std::cout << "  obs[" << i << "] = " << obs[i] << "\n";
    }
    std::cout << "  ... (total " << obs.size() << " dims)\n\n";

    // ランダムエージェントでシミュレーション
    std::cout << "--- ランダムエージェントによるシミュレーション ---\n";
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> action_dist(0, 2);

    bool done = false;
    int step_count = 0;
    double cumulative_reward = 0.0;

    while (!done && step_count < 20) { // 最初の20ステップだけ表示
        int action = action_dist(rng);
        auto result = env.step(action);
        cumulative_reward += result.reward;
        done = result.done;

        std::string action_str;
        switch(action) {
            case 0: action_str = "HOLD"; break;
            case 1: action_str = "BUY "; break;
            case 2: action_str = "SELL"; break;
        }

        std::cout << "Step " << std::setw(3) << step_count
                  << " | Action: " << action_str
                  << " | Reward: " << std::fixed << std::setprecision(4) << result.reward
                  << " | Portfolio: " << std::setprecision(2) << env.get_portfolio_value()
                  << "\n";
        step_count++;
    }

    // 残りのステップを一気に実行
    while (!done) {
        int action = action_dist(rng);
        auto result = env.step(action);
        cumulative_reward += result.reward;
        done = result.done;
    }

    std::cout << "\n累積報酬: " << cumulative_reward << "\n";

    // サマリー出力
    env.print_summary();

    // Buy & Hold 戦略との比較
    std::cout << "\n--- Buy & Hold 戦略との比較 ---\n";
    double buy_hold_shares = 100000.0 / prices[10]; // 最初のステップで全額購入
    double buy_hold_final = buy_hold_shares * prices.back();
    std::cout << "Buy & Hold 最終資産: " << buy_hold_final << "\n";
    std::cout << "Buy & Hold リターン: " << (buy_hold_final - 100000.0) / 100000.0 * 100.0 << "%\n";

    return 0;
}
