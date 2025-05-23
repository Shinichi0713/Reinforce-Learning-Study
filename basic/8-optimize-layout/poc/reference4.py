import numpy as np
import random, os
import matplotlib.pyplot as plt

# キャンバスサイズ
CANVAS_W, CANVAS_H = 300, 300
path_q_table = os.path.join(os.path.dirname(__file__), "Q_table.npy")

# 配置座標をGRID_STEPごとに離散化
GRID_STEP = 10
positions = []
for x in range(0, CANVAS_W, GRID_STEP):
    for y in range(0, CANVAS_H, GRID_STEP):
        positions.append((x, y))

# Qテーブル（[サイズカテゴリ][座標idx]）
# サイズカテゴリをいくつかに分割
size_categories = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100), (100, 120)]
Q = np.zeros((len(size_categories), len(positions)))
if os.path.exists(path_q_table):
    print("load q parameter")
    Q = np.load(path_q_table)

def is_overlap(placed, new_pos, new_size):
    nx, ny = new_pos
    nw, nh = new_size
    for (px, py, pw, ph) in placed:
        if not (nx + nw <= px or px + pw <= nx or ny + nh <= py or py + ph <= ny):
            return True
    return False

def get_size_category(w, h):
    for i, (min_s, max_s) in enumerate(size_categories):
        if min_s <= w <= max_s and min_s <= h <= max_s:
            return i
    return 0

def min_distance(placed, new_pos, new_size):
    nx, ny = new_pos
    nw, nh = new_size
    min_dist = np.inf
    for (px, py, pw, ph) in placed:
        # x方向の距離
        dx = max(px - (nx + nw), nx - (px + pw), 0)
        # y方向の距離
        dy = max(py - (ny + nh), ny - (py + ph), 0)
        dist = max(dx, dy)
        if dist < min_dist:
            min_dist = dist
    return min_dist if placed else np.inf

def give_reward(placed, rects):
    if not placed:
        return -1  # 配置失敗
    return 1  # 配置成功


def train_q_function():
    NUM_EPISODES = 1000
    ALPHA = 0.1
    GAMMA = 0.9
    EPSILON = 0.2

    best_layout = []
    best_count = 0

    for episode in range(NUM_EPISODES):
        print(f"Episode {episode + 1}/{NUM_EPISODES}")
        # ランダムな長方形リストを生成（個数もランダム）
        n_rects = random.randint(3, 12)
        rects = []
        for _ in range(n_rects):
            w = random.randint(20, 120)
            h = random.randint(20, 120)
            rects.append((w, h))

        placed = []
        used_idxs = set()
        count = 0
        reward_total = 0.0
        for step in range(len(rects)):
            candidates = [i for i in range(len(rects)) if i not in used_idxs]
            if not candidates:
                break
            rect_idx = random.choice(candidates)
            w, h = rects[rect_idx]
            cat = get_size_category(w, h)
            # ε-greedy
            if random.random() < EPSILON:
                pos_idx = random.randint(0, len(positions)-1)
            else:
                # サイズごとのQ値を取得
                qvals = Q[cat].copy()
                # 重なる場所は塞いでしまう
                for idx, pos in enumerate(positions):
                    if is_overlap(placed, pos, (w, h)):
                        qvals[idx] = -np.inf
                pos_idx = np.argmax(qvals)
                if qvals[pos_idx] == -np.inf:
                    break
            pos = positions[pos_idx]
            if is_overlap(placed, pos, (w, h)):
                reward = -10
                Q[cat, pos_idx] += ALPHA * (reward - Q[cat, pos_idx])
                break
            else:
                reward = 1
                dist = min_distance(placed, pos, (w, h))
                if dist <= 2:
                    reward += 2  # 密に詰めたら追加報酬
                Q[cat, pos_idx] += ALPHA * (reward + GAMMA * np.max(Q[cat]) - Q[cat, pos_idx])
                placed.append((pos[0], pos[1], w, h))
                used_idxs.add(rect_idx)
                count += 1
            reward_total += reward
        if count > best_count:
            best_count = count
            best_layout = placed.copy()

    print(f"最大配置数: {best_count}")
    np.save(path_q_table, Q)
    return best_layout

best_layout = train_q_function()

# 可視化
plt.figure(figsize=(6,6))
plt.imshow(np.ones((CANVAS_H, CANVAS_W, 3)))
for (x, y, w, h) in best_layout:
    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
    plt.gca().add_patch(rect)
plt.xlim(0, CANVAS_W)
plt.ylim(CANVAS_H, 0)
plt.title("layout sample (random rectangles)")
plt.show()
