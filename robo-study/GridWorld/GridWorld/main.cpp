#include "GridWorld.h"
#include "QAgent.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <tuple>


int main() {
    // 5x5 grid example
    GridWorld env(5, 5);
    // walls
    env.setWalls({ {1,1},{1,2},{2,1},{3,3} }); // example obstacles
    env.setStart(0, 0);
    env.setGoal(4, 4, 1.0);
    env.setStepPenalty(-0.02);

    int nS = env.nStates();
    int nA = env.nActions();

    QAgent agent(nS, nA, 0.3, 0.99, 0.4);

    const int episodes = 2000;
    const int max_steps = 200;

    for (int ep = 1; ep <= episodes; ++ep) {
        int s = env.reset();
        double total_reward = 0.0;
        bool done = false;
        for (int t = 0; t < max_steps; ++t) {
            int a = agent.selectAction(s);
            auto result = env.step(a);
            const int s2 = std::get<0>(result);
            const double r = std::get<1>(result);
            bool done2 = std::get<2>(result);
            auto [s2, r, done2] = env.step(a);
            agent.observe(s, a, r, s2, done2);
            s = s2;
            done = done2;
            total_reward += r;
            if (done) break;
        }

        // decay epsilon slowly
        if (ep % 50 == 0) agent.decayEpsilon(0.9);

        if (ep % 200 == 0 || ep == 1) {
            std::cout << "Episode " << ep << " total_reward=" << total_reward << " eps=" << std::fixed << std::setprecision(3) << agent.getQtable().size() /*dummy*/ << "\n";
        }
    }

    // show learned policy
    std::cout << "\nLearned policy (arrows). G=goal, #=wall\n";
    env.renderPolicy(agent.getQtable());

    // show Q-values for start state
    int start = env.reset();
    std::cout << "\nStart state: \n";
    env.renderState(start);

    return 0;
}
