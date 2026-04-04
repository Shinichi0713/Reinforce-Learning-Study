import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple

# ===== あなたの環境 =====
env = JaxCoopNavEnv()

# ===== ハイパーパラメータ =====
NUM_AGENTS = 2
OBS_DIM = 8
ACT_DIM = 5
STATE_DIM = 16

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 4
TARGET_UPDATE_FREQ = 10  # ターゲットネットワーク更新頻度

# ===== QMIX ネットワーク =====

def init_q_net(key, in_dim=OBS_DIM, hidden=64, out_dim=ACT_DIM):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (in_dim, hidden)) * 0.1,
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(k2, (hidden, out_dim)) * 0.1,
        "b2": jnp.zeros(out_dim),
    }

def q_net(params, obs):
    x = jnp.tanh(obs @ params["w1"] + params["b1"])
    return x @ params["w2"] + params["b2"]

def init_mixing_net(key, state_dim=STATE_DIM, hidden=32, num_agents=NUM_AGENTS):
    # ミキシングネットワーク（モノトニシティ制約付き）
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (state_dim, hidden)) * 0.1,
        "b1": jnp.zeros(hidden),
        "w2": jax.random.normal(k2, (hidden, num_agents)) * 0.1,  # 非負にする
        "b2": jnp.zeros(num_agents),
        "w_out": jax.random.normal(k3, (num_agents, 1)) * 0.1,   # 非負にする
        "b_out": jnp.zeros(1),
    }

def mixing_net(params, q_values, state):
    # q_values: [num_agents]
    # state: [state_dim]
    x = jnp.tanh(state @ params["w1"] + params["b1"])
    w2 = jax.nn.softplus(params["w2"])  # 非負
    w_out = jax.nn.softplus(params["w_out"])  # 非負
    # 線形結合（モノトニシティを保証）
    # q_values: [num_agents], w2: [hidden, num_agents] → [hidden]
    hidden = x @ w2 + params["b2"]  # [hidden]
    # hidden @ w_out: [hidden] @ [hidden, 1] → [1]
    q_total = hidden @ w_out + params["b_out"]
    return q_total[0]

# ===== パラメータ初期化 =====
key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)

# 各エージェントの Q ネットワーク（共有 or 個別）
q_params = {
    "agent_0": init_q_net(k1),
    "agent_1": init_q_net(k2),
}
mixing_params = init_mixing_net(k3)

# ターゲットネットワーク（コピー）
target_q_params = jax.tree.map(lambda x: x.copy(), q_params)
target_mixing_params = jax.tree.map(lambda x: x.copy(), mixing_params)

opt_q = optax.adam(LR)
opt_mix = optax.adam(LR)
opt_state_q = opt_q.init(q_params)
opt_state_mix = opt_mix.init(mixing_params)

# ===== 行動選択（ε-greedy） =====
def select_actions(q_params, obs, key, epsilon=0.1):
    actions = {}
    q_values = {}
    for i, agent in enumerate(["agent_0", "agent_1"]):
        q_vals = q_net(q_params[agent], obs[agent])
        q_values[agent] = q_vals

        # ε-greedy
        if jax.random.uniform(key, ()) < epsilon:
            action = jax.random.randint(key, (), 0, ACT_DIM)
        else:
            action = jnp.argmax(q_vals)
        actions[agent] = int(action)
    return actions, q_values

# ===== ロールアウト（QMIX 用に拡張） =====
def rollout(key, q_params, epsilon=0.1):
    obs, state = env.reset(key)
    traj = []

    for _ in range(env.max_steps):
        key, subkey = jax.random.split(key)
        actions, q_vals = select_actions(q_params, obs, subkey, epsilon)

        next_obs, next_state, rewards, done, _ = env.step(key, actions)

        traj.append({
            "obs": obs,
            "state": state,
            "actions": actions,
            "rewards": rewards,
            "q_values": q_vals,
            "done": done["__all__"]
        })

        obs, state = next_obs, next_state
        if done["__all__"]:
            break

    return traj

# ===== QMIX 損失関数 =====
def compute_qmix_loss(q_params, mixing_params, target_q_params, target_mixing_params, traj):
    # 1. 現在の Q_total を計算
    q_vals_list = []
    for step in traj:
        q_vals = []
        for agent in ["agent_0", "agent_1"]:
            q = q_net(q_params[agent], step["obs"][agent])
            q_vals.append(q[step["actions"][agent]])
        q_vals_list.append(jnp.array(q_vals))

    q_vals_arr = jnp.stack(q_vals_list)  # [T, num_agents]
    states = jnp.stack([step["state"] for step in traj])  # [T, state_dim]

    q_total = jax.vmap(mixing_net, in_axes=(None, 0, 0))(mixing_params, q_vals_arr, states)

    # 2. ターゲット Q_total を計算（TD(0)）
    target_q_vals_list = []
    for step in traj:
        q_vals = []
        for agent in ["agent_0", "agent_1"]:
            q = q_net(target_q_params[agent], step["obs"][agent])
            q_vals.append(jnp.max(q))  # ターゲットは max Q
        target_q_vals_list.append(jnp.array(q_vals))

    target_q_vals_arr = jnp.stack(target_q_vals_list)  # [T, num_agents]

    # 報酬の合計（グローバル報酬）
    rewards = jnp.array([step["rewards"]["agent_0"] + step["rewards"]["agent_1"] for step in traj])

    # TD ターゲット
    target_q_total = rewards + GAMMA * jax.vmap(mixing_net, in_axes=(None, 0, 0))(
        target_mixing_params, target_q_vals_arr, states
    )

    # 3. QMIX 損失（TD 誤差の二乗）
    loss = jnp.mean((q_total - target_q_total) ** 2)
    return loss

# ===== 更新（QMIX） =====
@jax.jit
def update(q_params, mixing_params, target_q_params, target_mixing_params,
           opt_state_q, opt_state_mix, traj):

    loss_fn = lambda qp, mp: compute_qmix_loss(
        qp, mp, target_q_params, target_mixing_params, traj
    )

    grads_q, grads_mix = jax.grad(loss_fn, argnums=(0, 1))(q_params, mixing_params)

    updates_q, opt_state_q = opt_q.update(grads_q, opt_state_q)
    updates_mix, opt_state_mix = opt_mix.update(grads_mix, opt_state_mix)

    q_params = optax.apply_updates(q_params, updates_q)
    mixing_params = optax.apply_updates(mixing_params, updates_mix)

    return q_params, mixing_params, opt_state_q, opt_state_mix

# ===== 学習ループ（QMIX） =====
key = jax.random.PRNGKey(42)

for episode in range(200):
    key, subkey = jax.random.split(key)
    epsilon = max(0.1, 1.0 - episode / 100)  # ε を徐々に減らす

    traj = rollout(subkey, q_params, epsilon=epsilon)

    q_params, mixing_params, opt_state_q, opt_state_mix = update(
        q_params, mixing_params, target_q_params, target_mixing_params,
        opt_state_q, opt_state_mix, traj
    )

    # ターゲットネットワークを更新（一定間隔で）
    if episode % TARGET_UPDATE_FREQ == 0:
        target_q_params = jax.tree.map(lambda x: x.copy(), q_params)
        target_mixing_params = jax.tree.map(lambda x: x.copy(), mixing_params)

    if episode % 10 == 0:
        total_reward = sum(
            step["rewards"]["agent_0"] + step["rewards"]["agent_1"]
            for step in traj
        )
        print(f"Episode {episode}, Reward: {total_reward:.2f}, epsilon: {epsilon:.2f}")