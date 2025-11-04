#include "QAgent.h"
#include <random>
#include <algorithm>
#include <chrono>

QAgent::QAgent(int n_states, int n_actions, double alpha, double gamma, double eps)
    : n_states_(n_states), n_actions_(n_actions),
    Q_(n_states* n_actions, 0.0),
    alpha_(alpha), gamma_(gamma), eps_(eps),
    uni_(0.0, 1.0),
    action_dist_(0, n_actions - 1)
{
    rng_.seed((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

int QAgent::selectAction(int state) {
    double roll = uni_(rng_);
    if (roll < eps_) {
        return action_dist_(rng_); // random
    }
    // greedy
    int best_a = 0;
    double best_q = Q_[state * n_actions_ + 0];
    for (int a = 1; a < n_actions_; ++a) {
        double q = Q_[state * n_actions_ + a];
        if (q > best_q) { best_q = q; best_a = a; }
    }
    return best_a;
}

void QAgent::observe(int s, int a, double r, int s2, bool done) {
    double q_sa = Q_[s * n_actions_ + a];
    double max_q_s2 = 0.0;
    if (!done) {
        max_q_s2 = Q_[s2 * n_actions_ + 0];
        for (int aa = 1; aa < n_actions_; ++aa) {
            max_q_s2 = std::max(max_q_s2, Q_[s2 * n_actions_ + aa]);
        }
    }
    double target = r + (done ? 0.0 : gamma_ * max_q_s2);
    Q_[s * n_actions_ + a] = q_sa + alpha_ * (target - q_sa);
}

void QAgent::decayEpsilon(double factor) {
    eps_ *= factor;
    if (eps_ < 0.01) eps_ = 0.01;
}
