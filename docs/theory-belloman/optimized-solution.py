import numpy as np
import copy

# MDPの定義
p = [0.8, 0.5, 1.0]
gamma = 0.95

# 報酬
r = np.zeros((3, 3, 2))
r[0, 1, 0] = 1.0
r[0, 2, 0] = 2.0
r[0, 0, 1] = 0
r[1, 0, 0] = 1.0
r[1, 2, 0] = 2.0
r[1, 1, 1] = 1.0
r[2, 0, 0] = 1.0
r[2, 1, 0] = 0.0
r[2, 2, 1] = -1.0


# 行動価値の初期化
v = [0, 0, 0]
v_prev = copy.copy(v)

# 行動価値関数の初期化
q = np.zeros((3, 2))

# 方策分布初期化
pi = [0.5, 0.5, 0.5]


for step in range(100):
    for i in range(3):
        q[i, 0] = p[i] * (r[i, (i+1)%3, 0] + gamma * v[(i+1)%3]) + \
            (1-p[i]) * (r[i, (i+2)%3, 0] + gamma*v[(i+2)%3])
        q[i, 1] = r[i, i, 1] + gamma * v[i]

        # 方策更新
        if q[i, 0] > q[i, 1]:
            pi[i] = 1
        elif q[i, 0] == q[i, 1]:
            pi[i] = 0.5
        else:
            pi[i] = 0

        v_new = np.max(q, axis=1)
        if np.min(v_new - v) <= 0:
            break
        v = copy.copy(v_new)
        print("step: {step}, v: {v}".format(step=step, v=v))

# 結果表示
print("converged step: {step}".format(step=step))
print("pi: {pi}".format(pi=pi))
print("v: {v}".format(v=v))
print("q: {q}".format(q=q))
