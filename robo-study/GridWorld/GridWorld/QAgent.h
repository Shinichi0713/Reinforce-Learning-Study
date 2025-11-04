#pragma once
#include <vector>

class QAgent {
public:
    QAgent(int n_states, int n_actions, double alpha = 0.1, double gamma = 0.99, double eps = 0.2);

    int selectAction(int state);
    void observe(int s, int a, double r, int s2, bool done);
    void decayEpsilon(double factor);

    const std::vector<double>& getQtable() const { return Q_; }

private:
    int n_states_, n_actions_;
    std::vector<double> Q_; // flattened n_states * n_actions: Q[s*n_actions + a]
    double alpha_, gamma_, eps_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uni_;
    std::uniform_int_distribution<int> action_dist_;
};
