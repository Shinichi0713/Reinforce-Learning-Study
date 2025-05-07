# 1次元迷路TD(0)学習サンプル

# 状態のリスト
states = ['S', 'A', 'G']  # S: スタート, A: 中間, G: ゴール

# 価値関数の初期化
V = {'S': 0.0, 'A': 0.0, 'G': 0.0}

# パラメータ
alpha = 0.5  # 学習率
gamma = 1.0  # 割引率

# 1エピソードのオンライン学習
trajectory = [('S', 'A', 0),  # (現状態, 次状態, 報酬)
              ('A', 'G', 1)]  # ゴール到達時のみ報酬1

print("初期値:", V)

for (s, s_next, r) in trajectory:
    # TD誤差の計算
    td_error = r + gamma * V[s_next] - V[s]
    # 学習
    V[s] += alpha * td_error
    # 状態ごとの値を表示
    print(f"{s} → {s_next}（報酬:{r}）: TD誤差={td_error:.2f}, 新しいV({s})={V[s]:.2f}")

print("エピソード後の価値関数:", V)
