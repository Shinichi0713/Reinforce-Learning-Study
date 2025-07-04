import numpy as np
import gym

# 環境の初期化
env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset()
n_states = env.observation_space.n
n_actions = env.action_space.n

# 特徴量関数（one-hotエンコーディング）
def feature_matrix():
    return np.eye(n_states)

F = feature_matrix()

# 専門家デモデータの生成（ここでは環境の最適方策を利用）
def generate_expert_trajectories(env, policy, n_trajs=20, max_steps=20):
    trajs = []
    for _ in range(n_trajs):
        s, _ = env.reset()  # ←ここを修正
        traj = []
        for _ in range(max_steps):
            a = policy[s]
            s, r, terminated, truncated, _ = env.step(a)  # ←ここも修正
            traj.append((s, a))
            if terminated or truncated:
                break
        trajs.append(traj)
    return trajs

# 簡易な最適方策を事前計算（FrozenLakeは小さいので全探索）
def value_iteration(env, gamma=0.9, eps=1e-5):
    V = np.zeros(n_states)
    while True:
        prev_V = V.copy()
        for s in range(n_states):
            Qs = []
            for a in range(n_actions):
                env.env.s = s
                s_, r, done, _, _ = env.step(a)
                Qs.append(r + gamma * V[s_])
            V[s] = max(Qs)
        if np.max(np.abs(V - prev_V)) < eps:
            break
    # 最適方策
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        Qs = []
        for a in range(n_actions):
            env.env.s = s
            s_, r, done, _, _ = env.step(a)
            Qs.append(r + gamma * V[s_])
        policy[s] = np.argmax(Qs)
    return policy

# 専門家デモ生成
expert_policy = value_iteration(env)
expert_trajs = generate_expert_trajectories(env, expert_policy, n_trajs=30)

# 専門家特徴量期待値
def compute_feature_expectations(trajs, F):
    feat_exp = np.zeros(F.shape[1])
    for traj in trajs:
        for (s, a) in traj:
            feat_exp += F[s]
    return feat_exp / len(trajs)

expert_feat_exp = compute_feature_expectations(expert_trajs, F)

# 報酬パラメータ初期化
w = np.random.randn(n_states)

# MaxEnt IRLのメインループ
def soft_value_iteration(w, F, env, gamma=0.9, eps=1e-4):
    R = F @ w
    V = np.zeros(n_states)
    while True:
        prev_V = V.copy()
        for s in range(n_states):
            Qs = []
            for a in range(n_actions):
                env.env.s = s
                s_, r, done, _, _ = env.step(a)
                Qs.append(R[s] + gamma * V[s_])
            # softmax backup
            V[s] = np.log(np.sum(np.exp(Qs)))
        if np.max(np.abs(V - prev_V)) < eps:
            break
    # softmax方策
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        Qs = []
        for a in range(n_actions):
            env.env.s = s
            s_, r, done, _, _ = env.step(a)
            Qs.append(R[s] + gamma * V[s_])
        Qs = np.array(Qs)
        policy[s] = np.exp(Qs - np.max(Qs))  # for numerical stability
        policy[s] /= np.sum(policy[s])
    return policy

def generate_expert_trajectories(env, policy, n_trajs=20, max_steps=20):
    trajs = []
    for _ in range(n_trajs):
        s, _ = env.reset()  # ← sだけ取り出す
        traj = []
        for _ in range(max_steps):
            a = policy[s]
            s_, r, terminated, truncated, _ = env.step(a)
            traj.append((s, a))
            s = s_  # ← 状態を更新
            if terminated or truncated:
                break
        trajs.append(traj)
    return trajs

def sample_trajectories(env, policy, n_trajs=20, max_steps=20):
    trajs = []
    for _ in range(n_trajs):
        s, _ = env.reset()  # ← sだけ取り出す
        traj = []
        for _ in range(max_steps):
            a = np.random.choice(n_actions, p=policy[s])
            s_, r, terminated, truncated, _ = env.step(a)
            traj.append((s, a))
            s = s_  # ← 状態を更新
            if terminated or truncated:
                break
        trajs.append(traj)
    return trajs


# IRL学習ループ
lr = 0.1
for it in range(30):
    # softmax方策で軌跡をサンプリング
    policy = soft_value_iteration(w, F, env)
    agent_trajs = sample_trajectories(env, policy)
    agent_feat_exp = compute_feature_expectations(agent_trajs, F)
    grad = expert_feat_exp - agent_feat_exp
    w += lr * grad
    print(f"Iter {it}: ||grad||={np.linalg.norm(grad):.4f}")

# 学習した報酬
import matplotlib.pyplot as plt

print("Learned reward:", w.reshape(4,4))
# --- 結果の可視化 ---
plt.imshow(w.reshape(4,4))
plt.colorbar()
plt.title("Recovered Reward")
plt.show()

"""
Learned reward: [[49.52453723 -5.04453093 -1.28035062 -2.0666169 ]
 [-1.77078926  0.21384259 -1.00230605  0.22454173]
 [ 0.44180976 -0.88529756  0.16931089  0.08238014]
 [-0.91228947  0.22185799 -1.26040765 -0.94330021]]"""