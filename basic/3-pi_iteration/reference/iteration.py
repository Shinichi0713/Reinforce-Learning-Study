import numpy as np

# --- 環境クラス ---
class GridWorldEnv:
    def __init__(self):
        self.states = [1, 2, 3, 4]
        self.actions = ['左', '右']
        self.gamma = 1.0
        self.terminal_state = 4

    def next_state(self, s, a):
        if s == 1:
            return 1 if a == '左' else 2
        elif s == 2:
            return 1 if a == '左' else 3
        elif s == 3:
            return 2 if a == '左' else 4
        elif s == 4:
            return 4

    def reward(self, s, a, s_next):
        return 1 if s_next == 4 and s != 4 else 0

    def is_terminal(self, s):
        return s == self.terminal_state

# --- エージェントクラス ---
class PolicyIterationAgent:
    def __init__(self, env):
        self.env = env
        self.policy = {s: '右' for s in env.states}
        self.V = {s: 0.0 for s in env.states}

    def policy_evaluation(self, threshold=1e-8):
        while True:
            delta = 0
            for s in self.env.states:
                if self.env.is_terminal(s):
                    continue
                a = self.policy[s]
                s_next = self.env.next_state(s, a)
                r = self.env.reward(s, a, s_next)
                v_new = r + self.env.gamma * self.V[s_next]
                delta = max(delta, abs(self.V[s] - v_new))
                self.V[s] = v_new
            if delta < threshold:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in self.env.states:
            if self.env.is_terminal(s):
                continue
            old_action = self.policy[s]
            action_values = {}
            for a in self.env.actions:
                s_next = self.env.next_state(s, a)
                r = self.env.reward(s, a, s_next)
                action_values[a] = r + self.env.gamma * self.V[s_next]
            best_action = max(action_values, key=action_values.get)
            self.policy[s] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

# --- 方策反復プログラム ---
def policy_iteration(env):
    agent = PolicyIterationAgent(env)
    iteration = 0
    while True:
        agent.policy_evaluation()
        policy_stable = agent.policy_improvement()
        iteration += 1
        print(f"Iteration {iteration}:")
        print("  V =", {k: round(v, 3) for k, v in agent.V.items()})
        print("  policy =", agent.policy)
        if policy_stable:
            break
    print("\n最適方策:", agent.policy)
    print("最適状態価値:", {k: round(v, 3) for k, v in agent.V.items()})
    return agent.policy, agent.V

# --- 実行 ---
if __name__ == "__main__":
    env = GridWorldEnv()
    policy, V = policy_iteration(env)
