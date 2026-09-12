%%writefile maze_rl.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>

// 5x5 の迷路マップ (0: 通路, 1: 壁, 2: スタート, 3: ゴール)
const int HEIGHT = 5;
const int WIDTH = 5;
const int maze[HEIGHT][WIDTH] = {
    {2, 0, 1, 0, 0},
    {0, 1, 0, 0, 1},
    {0, 0, 0, 1, 0},
    {1, 1, 0, 1, 0},
    {0, 0, 0, 0, 3}
};

// 上, 下, 左, 右
const int dx[] = {-1, 1, 0, 0};
const int dy[] = {0, 0, -1, 1};

int main() {
    // 状態数 = HEIGHT * WIDTH, 行動数 = 4
    std::vector<std::vector<double>> Q(HEIGHT * WIDTH, std::vector<double>(4, 0.0));

    double alpha = 0.1, gamma = 0.9, epsilon = 0.2;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // --- Q学習 ---
    for (int ep = 0; ep < 500; ++ep) {
        int r = 0, c = 0; // スタート位置 (0,0)
        int steps = 0;

        while ((r != 4 || c != 4) && steps < 100) {
            int state = r * WIDTH + c;
            int action;

            if (dis(gen) < epsilon) {
                action = std::uniform_int_distribution<>(0, 3)(gen);
            } else {
                action = std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
            }

            int nr = r + dx[action];
            int nc = c + dy[action];

            // 移動範囲と壁のチェック
            double reward;
            if (nr < 0 || nr >= HEIGHT || nc < 0 || nc >= WIDTH || maze[nr][nc] == 1) {
                nr = r;
                nc = c;
                reward = -0.5; // 壁衝突ペナルティ
            } else if (maze[nr][nc] == 3) {
                reward = 10.0; // ゴール報酬
            } else {
                reward = -0.04; // ステップペナルティ
            }

            int next_state = nr * WIDTH + nc;
            double max_next_q = *std::max_element(Q[next_state].begin(), Q[next_state].end());
            Q[state][action] += alpha * (reward + gamma * max_next_q - Q[state][action]);

            r = nr;
            c = nc;
            steps++;
        }
    }

    // --- 学習済みモデルで評価走行（ログ保存） ---
    std::ofstream traj_file("trajectory.csv");
    traj_file << "row,col\n";

    int r = 0, c = 0;
    traj_file << r << "," << c << "\n";

    int eval_steps = 0;
    while ((r != 4 || c != 4) && eval_steps < 50) {
        int state = r * WIDTH + c;
        // 最善の行動を選択 (決定論的選択)
        int action = std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();

        int nr = r + dx[action];
        int nc = c + dy[action];

        if (nr >= 0 && nr < HEIGHT && nc >= 0 && nc < WIDTH && maze[nr][nc] != 1) {
            r = nr;
            c = nc;
        }
        traj_file << r << "," << c << "\n";
        eval_steps++;
    }
    traj_file.close();

    std::cout << "Evaluation trajectory saved successfully." << std::endl;
    return 0;
}

