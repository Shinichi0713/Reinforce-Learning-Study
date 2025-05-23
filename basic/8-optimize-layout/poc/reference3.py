import numpy as np
import random
import matplotlib.pyplot as plt

# キャンバスサイズ
CANVAS_W, CANVAS_H = 300, 300

# 画像リスト（幅, 高さ）
images = [(60, 80), (30, 40), (100, 50), (40, 90), (80, 30)]

# 配置可能な座標をGRID_STEPごとに離散化
GRID_STEP = 10
positions = []
for x in range(0, CANVAS_W, GRID_STEP):
    for y in range(0, CANVAS_H, GRID_STEP):
        positions.append((x, y))

# Qテーブル（[画像idx][座標idx]）
Q = np.zeros((len(images), len(positions)))

def is_overlap(placed, new_pos, new_size):
    nx, ny = new_pos
    nw, nh = new_size
    for (px, py, pw, ph) in placed:
        if not (nx + nw <= px or px + pw <= nx or ny + nh <= py or py + ph <= ny):
            return True
    return False

NUM_EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

best_layout = []
best_count = 0

for episode in range(NUM_EPISODES):
    placed = []
    used_imgs = set()
    count = 0
    for step in range(len(images)):
        # まだ配置していない画像
        candidates = [i for i in range(len(images)) if i not in used_imgs]
        if not candidates:
            break
        img_idx = random.choice(candidates)
        img_size = images[img_idx]
        # ε-greedyで配置座標選択
        if random.random() < EPSILON:
            pos_idx = random.randint(0, len(positions)-1)
        else:
            qvals = Q[img_idx].copy()
            # 重なる座標は選ばない
            for idx, pos in enumerate(positions):
                if is_overlap(placed, pos, img_size):
                    qvals[idx] = -np.inf
            pos_idx = np.argmax(qvals)
            if qvals[pos_idx] == -np.inf:
                break
        pos = positions[pos_idx]
        if is_overlap(placed, pos, img_size):
            reward = -10  # 大きなペナルティ
            Q[img_idx, pos_idx] += ALPHA * (reward - Q[img_idx, pos_idx])
            break
        else:
            reward = 1
            Q[img_idx, pos_idx] += ALPHA * (reward + GAMMA * np.max(Q[img_idx]) - Q[img_idx, pos_idx])
            placed.append((pos[0], pos[1], img_size[0], img_size[1]))
            used_imgs.add(img_idx)
            count += 1
    if count > best_count:
        best_count = count
        best_layout = placed.copy()

print(f"最大配置数: {best_count}")

# 可視化
plt.figure(figsize=(6,6))
plt.imshow(np.ones((CANVAS_H, CANVAS_W, 3)))
for (x, y, w, h) in best_layout:
    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='none')
    plt.gca().add_patch(rect)
plt.xlim(0, CANVAS_W)
plt.ylim(CANVAS_H, 0)
plt.title("layout sample (various sizes)")
plt.show()
