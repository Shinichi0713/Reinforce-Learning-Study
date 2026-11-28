"""
overview:
    方策反復法により３状態２行動のマルコフ決定過程を解く

args:
    各種パラメータ設定値は、本コード中に明記される

output:
    反復ステップごとに３状態の価値関数値および方策確率を出力

usage-example:
    python3 policy_iteraton.py
"""
# 各種モジュールのインポート
import numpy as np
import copy

# MDP の設定
p = [0.8, 0.5, 1.0]

# 割引率の設定
gamma = 0.95

# 報酬期待値の設定
r = np.zeros((3, 3, 2))
r[0, 1, 0] = 1.0
r[0, 2, 0] = 2.0
r[0, 0, 1] = 0.0
r[1, 0, 0] = 1.0
r[1, 2, 0] = 2.0
r[1, 1, 1] = 1.0
r[2, 0, 0] = 1.0
r[2, 1, 0] = 0.0
r[2, 2, 1] = -1.0

# 価値関数の初期化
v = [0, 0, 0]
v_prev = copy.copy(v)

# 行動価値関数の初期化
q = np.zeros((3, 2))

# 方策分布の初期化
pi = [0.5, 0.5, 0.5]


# 方策評価関数の定義
def policy_estimator(pi, p, r, gamma):
    # 初期化
    R = [0, 0, 0]
    P = np.zeros((3, 3))
    A = np.zeros((3, 3))

    for i in range(3):

        # 状態遷移行列の計算
        P[i, i] = 1 - pi[i]
        P[i, (i + 1) % 3] = p[i] * pi[i]
        P[i, (i + 2) % 3] = (1 - p[i]) * pi[i]

        # 報酬ベクトルの計算
        R[i] = pi[i] * (p[i] * r[i, (i + 1) % 3, 0] +
                        (1 - p[i]) * r[i, (i + 2) % 3, 0]
                        ) + (1 - pi[i]) * r[i, i, 1]

    # 行列計算によるベルマン方程式の求解
    A = np.eye(3) - gamma * P
    B = np.linalg.inv(A)
    v_sol = np.dot(B, R)

    return v_sol


# 方策反復法の計算
for step in range(100):

    # 方策評価ステップ
    v = policy_estimator(pi, p, r, gamma)

    # 価値関数 v が前ステップの値 v_prep を改善しなければ終了
    if np.min(v - v_prev) <= 0:
        break

    # 現ステップの価値関数と方策を表示
    print('step:', step, ' value:', v, ' policy:', pi)

    # 方策改善ステップ
    for i in range(3):

        # 行動価値関数を計算
        q[i, 0] = p[i] * (
            r[i, (i + 1) % 3, 0] + gamma * v[(i + 1) % 3]
        ) + (1 - p[i]) * (r[i, (i + 2) % 3, 0]
                          + gamma * v[(i + 2) % 3])
        q[i, 1] = r[i, i, 1] + gamma * v[i]

        # 行動価値関数のもとで greedy に方策を改善
        if q[i, 0] > q[i, 1]:
            pi[i] = 1
        elif q[i, 0] == q[i, 1]:
            pi[i] = 0.5
        else:
            pi[i] = 0

    # 現ステップの価値関数を記録
    v_prev = copy.copy(v)
