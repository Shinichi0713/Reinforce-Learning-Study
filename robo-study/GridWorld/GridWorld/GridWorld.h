#pragma once
#include <vector>
#include <string>

class GridWorld {
public:
    // actions: 0=up,1=right,2=down,3=left
    enum Action { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

    GridWorld(int rows, int cols);
    void setWalls(const std::vector<std::pair<int, int>>& walls);
    void setStart(int r, int c);
    void setGoal(int r, int c, double reward = 1.0);
    void setStepPenalty(double p);
    int reset(); // return start state id
    // step: perform action, returns (next_state, reward, done)
    std::tuple<int, double, bool> step(int action);
    void renderPolicy(const std::vector<double>& Qtable) const; // expects Qtable size = n_states * 4
    void renderState(int state) const;
    int nStates() const { return rows_ * cols_; }
    int nActions() const { return 4; }
    int stateId(int r, int c) const { return r * cols_ + c; }
    std::pair<int, int> idToCoord(int id) const { return { id / cols_, id % cols_ }; }
    std::tuple<int, double, bool> step(int action);

private:
    int rows_, cols_;
    std::vector<int> grid_; // 0=free, 1=wall
    int start_r_, start_c_;
    int goal_r_, goal_c_;
    double goal_reward_;
    double step_penalty_;
    int cur_r_, cur_c_;

    bool isValid(int r, int c) const;
    
};
