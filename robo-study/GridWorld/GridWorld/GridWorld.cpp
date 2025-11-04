#include "GridWorld.h"
#include <iostream>
#include <iomanip>
#include <tuple>

GridWorld::GridWorld(int rows, int cols)
    : rows_(rows), cols_(cols), grid_(rows* cols, 0),
    start_r_(0), start_c_(0), goal_r_(rows - 1), goal_c_(cols - 1),
    goal_reward_(1.0), step_penalty_(-0.01), cur_r_(0), cur_c_(0) {}

void GridWorld::setWalls(const std::vector<std::pair<int, int>>& walls) {
    for (auto& p : walls) {
        if (isValid(p.first, p.second)) {
            grid_[stateId(p.first, p.second)] = 1;
        }
    }
}

void GridWorld::setStart(int r, int c) { start_r_ = r; start_c_ = c; }
void GridWorld::setGoal(int r, int c, double reward) { goal_r_ = r; goal_c_ = c; goal_reward_ = reward; }
void GridWorld::setStepPenalty(double p) { step_penalty_ = p; }

bool GridWorld::isValid(int r, int c) const {
    return r >= 0 && r < rows_&& c >= 0 && c < cols_&& grid_[stateId(r, c)] == 0;
}

int GridWorld::reset() {
    cur_r_ = start_r_;
    cur_c_ = start_c_;
    return stateId(cur_r_, cur_c_);
}

std::tuple<int, double, bool> GridWorld::step(int action) {
    int nr = cur_r_;
    int nc = cur_c_;
    if (action == UP) nr--;
    else if (action == RIGHT) nc++;
    else if (action == DOWN) nr++;
    else if (action == LEFT) nc--;

    if (!isValid(nr, nc)) {
        // invalid move -> stay and small penalty
        double r = step_penalty_ - 0.1; // bump penalty for hitting wall
        return { stateId(cur_r_, cur_c_), r, false };
    }

    cur_r_ = nr; cur_c_ = nc;
    if (cur_r_ == goal_r_ && cur_c_ == goal_c_) {
        return { stateId(cur_r_, cur_c_), goal_reward_, true };
    }
    else {
        return { stateId(cur_r_, cur_c_), step_penalty_, false };
    }
}

void GridWorld::renderState(int state) const {
    auto [r, c] = idToCoord(state);
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            if (i == r && j == c) std::cout << " A ";
            else if (i == start_r_ && j == start_c_) std::cout << " S ";
            else if (i == goal_r_ && j == goal_c_) std::cout << " G ";
            else if (grid_[stateId(i, j)] == 1) std::cout << " # ";
            else std::cout << " . ";
        }
        std::cout << "\n";
    }
}

void GridWorld::renderPolicy(const std::vector<double>& Qtable) const {
    // Qtable: [state * 4 + action]
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            int id = stateId(i, j);
            if (grid_[id] == 1) {
                std::cout << " # ";
                continue;
            }
            if (i == goal_r_ && j == goal_c_) { std::cout << " G "; continue; }
            int best_a = 0;
            double best_q = Qtable[id * 4 + 0];
            for (int a = 1; a < 4; ++a) {
                double q = Qtable[id * 4 + a];
                if (q > best_q) { best_q = q; best_a = a; }
            }
            const char* arrow = "?";
            if (best_a == UP) arrow = "Å™";
            if (best_a == RIGHT) arrow = "Å®";
            if (best_a == DOWN) arrow = "Å´";
            if (best_a == LEFT) arrow = "Å©";
            std::cout << " " << arrow << " ";
        }
        std::cout << "\n";
    }
}
