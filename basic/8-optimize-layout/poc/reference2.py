import numpy as np
import random
import matplotlib.pyplot as plt

# 設定
CANVAS_W, CANVAS_H = 500, 500
IMG_W, IMG_H = 50, 50
N_IMAGES = 20
GRID_STEP = 10  # 配置位置の間隔（ピクセル）

# 配置可能な座標リスト作成
positions = []
for x in range(0, CANVAS_W - IMG_W + 1, GRID_STEP):
    for y in range(0, CANVAS_H - IMG_H + 1, GRID_STEP):
        positions.append((x, y))

# Qテーブル（状態は未考慮、行動は配置座標のインデックス）
Q = np.zeros(len(positions))

def is_overlap(pos_list, new_pos):
    # 既存配置と重なるか判定
    for (x, y) in pos_list:
        if not (new_pos[0] + IMG_W <= x or x + IMG_W <= new_pos[0] or
                new_pos[1] + IMG_H <= y or y + IMG_H <= new_pos[1]):
            return True
    return False

NUM_EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

best_layout = []
best_count = 0

for episode in range(NUM_EPISODES):
    used_pos = []
    count = 0
    for i in range(N_IMAGES):
        # ε-greedy
        if random.random() < EPSILON:
            act_idx = random.randint(0, len(positions)-1)
        else:
            # 使っていない位置から最大Q値を選択
            qvals = Q.copy()
            for idx, pos in enumerate(positions):
                if is_overlap(used_pos, pos):
                    qvals[idx] = -np.inf
            act_idx = np.argmax(qvals)
            if qvals[act_idx] == -np.inf:
                break  # もう置けない
        pos = positions[act_idx]
        if is_overlap(used_pos, pos):
            reward = -1  # 重なったらペナルティ
            Q[act_idx] += ALPHA * (reward - Q[act_idx])
            break
        else:
            reward = 1  # うまく配置できた
            Q[act_idx] += ALPHA * (reward + GAMMA * np.max(Q) - Q[act_idx])
            used_pos.append(pos)
            count += 1
    if count > best_count:
        best_count = count
        best_layout = used_pos.copy()

print(f"最大配置数: {best_count}")

# 結果を可視化
plt.figure(figsize=(6,6))
plt.imshow(np.ones((CANVAS_H, CANVAS_W, 3)))  # 白背景
for (x, y) in best_layout:
    rect = plt.Rectangle((x, y), IMG_W, IMG_H, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
plt.xlim(0, CANVAS_W)
plt.ylim(CANVAS_H, 0)
plt.title("最適な敷き詰め結果")
plt.show()
