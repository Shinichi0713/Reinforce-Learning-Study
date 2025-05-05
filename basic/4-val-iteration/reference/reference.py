import numpy as np

# 状態と行動の定義
states = [1, 2, 3, 4]
actions = ['左', '右']
gamma = 1.0

# 状態遷移と報酬の定義
def step(state, action):
    if state == 4:
        return 4, 0  # ゴール状態は何をしても状態4にとどまる
    if action == '左':
        next_state = max(1, state - 1)
    else:
        next_state = min(4, state + 1)
    reward = 1 if next_state == 4 and state != 4 else 0
    return next_state, reward

# 価値反復法の実装
V = np.zeros(len(states))  # 状態価値関数
theta = 1e-6               # 収束判定用の閾値

while True:
    delta = 0
    for i, state in enumerate(states):
        if state == 4:
            continue  # ゴール状態は固定
        v = V[i]
        values = []
        for action in actions:
            next_state, reward = step(state, action)
            next_index = states.index(next_state)
            values.append(reward + gamma * V[next_index])
        V[i] = max(values)
        delta = max(delta, abs(v - V[i]))
    if delta < theta:
        break

# 最適方策の導出
policy = []
for i, state in enumerate(states):
    if state == 4:
        policy.append('-')  # ゴール状態では行動不要
        continue
    action_values = []
    for action in actions:
        next_state, reward = step(state, action)
        next_index = states.index(next_state)
        action_values.append(reward + gamma * V[next_index])
    best_action = actions[np.argmax(action_values)]
    policy.append(best_action)

# 結果表示
print("状態価値関数 V:")
for s, v in zip(states, V):
    print(f"  状態{s}: {v:.4f}")

print("\n最適方策:")
for s, a in zip(states, policy):
    print(f"  状態{s}: {a}")
