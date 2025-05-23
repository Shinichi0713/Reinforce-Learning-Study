import numpy as np
import random, os
import matplotlib.pyplot as plt

CANVAS_W, CANVAS_H = 300, 300
GRID_STEP = 10
positions = [(x, y) for x in range(0, CANVAS_W, GRID_STEP) for y in range(0, CANVAS_H, GRID_STEP)]
path_q_table = os.path.join(os.path.dirname(__file__), "Q_table.npy")

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

    dx1_max = np.inf
    dx2_max = np.inf
    dy1_max = np.inf
    dy2_max = np.inf
    for (px, py, pw, ph) in placed:
        # x方向の距離
        dx1_max = min(dx1_max, px - (nx + nw))
        dx2_max = min(dx2_max, (nx - (px + pw)))
        # y方向の距離
        dy1_max = min(dy1_max, py - (ny + nh))
        dy2_max = min(dy2_max, (ny - (py + ph)))
        dist = max(dx1_max, dx2_max, dy1_max, dy2_max)
        if dist < min_dist:
            min_dist = dist
    return min_dist if placed else np.inf

NUM_EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

best_layout = []
best_count = 0

for episode in range(NUM_EPISODES):
    n_rects = random.randint(3, 14)
    rects = [(random.randint(20, 120), random.randint(20, 120)) for _ in range(n_rects)]
    placed = []
    used_idxs = set()
    count = 0
    for step in range(len(rects)):
        candidates = [i for i in range(len(rects)) if i not in used_idxs]
        if not candidates:
            break
        rect_idx = random.choice(candidates)
        w, h = rects[rect_idx]
        cat = get_size_category(w, h)
        if random.random() < EPSILON:
            pos_idx = random.randint(0, len(positions)-1)
        else:
            qvals = Q[cat].copy()
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
    if count > best_count:
        best_count = count
        best_layout = placed.copy()
np.save(path_q_table, Q)
print(f"最大配置数: {best_count}")

plt.figure(figsize=(6,6))
plt.imshow(np.ones((CANVAS_H, CANVAS_W, 3)))
for (x, y, w, h) in best_layout:
    rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
    plt.gca().add_patch(rect)
plt.xlim(0, CANVAS_W)
plt.ylim(CANVAS_H, 0)
plt.title("layout sample (random rectangles)")
plt.show()
