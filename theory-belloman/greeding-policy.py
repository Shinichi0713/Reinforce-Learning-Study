
import numpy as np
import copy

'''
シチュエーション：
状態遷移のMDPが分かっているケース
価値関数の更新を行い、方策を更新する
コードの狙い：
今回の報酬に合うように方策を更新する
という一連のベルマン方程式のシミュレーションを行う
'''


# MDPの定義
p = [0.8, 0.5, 1.0]     
gamma = 0.95

# 報酬
r = np.zeros((3, 3, 2))
# r[0, 1, 0] = 1.0
# r[0, 2, 0] = 1.0
# r[0, 0, 1] = 4.0
# r[1, 0, 0] = 5.0
# r[1, 2, 0] = -1.0
# r[1, 1, 1] = -1.0
# r[2, 0, 0] = 3.0
# r[2, 1, 0] = 0.0
# r[2, 2, 1] = -1.0
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

# 方策評価関数
def policy_evaluation(pi, p, r, gamma):
    # init
    R = [0, 0, 0]
    P = np.zeros((3, 3))
    A = np.zeros((3, 3))

    for i in range(3):
        P[i, i] = 1 - pi[i]
        P[i, (i + 1) % 3] = p[i] * pi[i]
        P[i, (i + 2) % 3] = (1 - p[i]) * pi[i]

        # 報酬ベクトル
        R[i] = pi[i] * (p[i] * r[i, (i+1)%3,0]+ \
                (1-pi[i]) * r[i, (i+2)%3,0]) + \
                (1-pi[i]) * r[i, i, 1]
        
        # ベルマンの解
        A = np.eye(3) - gamma * P
        B = np.linalg.inv(A)
        v_sol = np.dot(B, R)
    return v_sol

for step in range(100):
    # 方策評価ステップ
    v = policy_evaluation(pi, p, r, gamma)
    
    print("step: {step}, v: {v}".format(step=step, v=v))

    for i in range(3):
        q[i, 0] = p[i] * (r[i, (i+1)%3, 0] + gamma * v[(i+1)%3]) + \
            (1-p[i]) * (r[i, (i+2)%3, 0] + gamma*v[(i+2)%3])
        q[i, 1] = r[i, i, 1] + gamma * v[i]

        if q[i, 0] > q[i, 1]:
            pi[i] = 1
        elif q[i, 0] == q[i, 1]:
            pi[i] = 0.5
        else:
            pi[i] = 0
    print("step: {step}, pi: {pi}".format(step=step, pi=pi))
    # 価値関数vを確認して収束判断
    if np.min(v - v_prev) <= 0:
        break
    # 現ステップ
    v_prev = copy.copy(v)

# 結果表示
print("converged step: {step}".format(step=step))
print("pi: {pi}".format(pi=pi))
print("v: {v}".format(v=v))
print("q: {q}".format(q=q))




