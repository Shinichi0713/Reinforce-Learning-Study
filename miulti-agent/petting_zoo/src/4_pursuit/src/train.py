import torch
import numpy as np
from collections import deque

# ハイパーパラメータ
max_episodes = 1000
max_cycles = 500
batch_size = 64
update_epochs = 3
gamma = 0.99
gae_lambda = 0.95
checkpoint_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 環境とラッパの初期化
env = PursuitWrapper(render_mode=None, max_cycles=max_cycles)
num_agents = env.num_agents
obs_dim = env.obs_dim
state_dim = env.state_dim
action_dim = env.action_dim

# MAPPOの初期化（CNNベース） → deviceを渡す
mappo = MAPPO(
    num_agents=num_agents,
    obs_dim=obs_dim,
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=gamma,
    gae_lambda=gae_lambda,
    device=device  # 追加: モデルを指定デバイスに移動
)

print("=== MAPPO（CNNベース）学習開始 ===")
print(f"エージェント数: {num_agents}")
print(f"観測次元: {obs_dim}")
print(f"グローバル状態次元: {state_dim}")
print(f"行動空間サイズ: {action_dim}")
print(f"使用デバイス: {device}")

path_checkpoint = os.path.join(CHECKPOINT_DIR, "mappo_episode.pth")
episode = mappo.load_checkpoint(path_checkpoint)
print(f"再開エピソード: {episode}")

# モニタリング用のバッファ
reward_buffer = deque(maxlen=100)
capture_buffer = deque(maxlen=100)  # 1エピソードあたりの捕獲数

for episode in range(max_episodes):
    # 環境リセット
    env.reset()
    episode_reward = 0.0
    episode_captures = 0  # このエピソードの捕獲数

    # 1エピソード実行
    for agent in env.env.agent_iter():
        # 観測とグローバル状態を取得
        obs = env.get_obs(agent)
        global_state = env.get_global_state()

        if agent not in env.env.agents:
            action = None
            reward = 0.0
            terminated = True
            truncated = True
        else:
            _, _, terminated, truncated, _ = env.env.last(agent)
            if terminated or truncated:
                action = None
            else:
                # 各エージェントの観測をテンソルにまとめ、deviceに送る
                obs_list = []
                for i in range(num_agents):
                    agent_name = f'pursuer_{i}'
                    if agent_name in env.env.agents:
                        agent_obs = env.get_obs(agent_name)
                        if agent_obs is not None:
                            obs_list.append(agent_obs)
                    else:
                        obs_list.append(np.zeros(obs_dim, dtype=np.float32))

                obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)  # (num_agents, obs_dim)
                global_state_tensor = torch.FloatTensor(global_state).to(device)  # (state_dim,)

                # MAPPOで行動を選択
                actions_np, log_probs_np = mappo.get_action(obs_tensor)

                # 現在のエージェントの行動を取得
                agent_idx = int(agent.split('_')[-1])
                action = actions_np[agent_idx]

            # 1ステップ進める
            reward, terminated, truncated, info = env.step(agent, action)
            episode_reward += reward

            # 報酬が +5 なら捕獲発生
            if reward == 5.0:
                episode_captures += 1

        # 各エージェントの観測・行動・報酬・log_probを辞書でまとめる
        obs_dict = {}
        action_dict = {}
        reward_dict = {}
        log_prob_dict = {}

        for i in range(num_agents):
            agent_name = f'pursuer_{i}'
            if agent_name in env.env.agents:
                agent_obs = env.get_obs(agent_name)
                if agent_obs is not None:
                    obs_dict[agent_name] = agent_obs
                else:
                    obs_dict[agent_name] = np.zeros(obs_dim, dtype=np.float32)

                if agent_name == agent:
                    action_dict[agent_name] = action if action is not None else 0
                    reward_dict[agent_name] = reward
                else:
                    action_dict[agent_name] = 0
                    reward_dict[agent_name] = 0.0

                # log_probはMAPPOから取得した値を使う
                log_prob_dict[agent_name] = log_probs_np[i] if 'log_probs_np' in locals() else 0.0
            else:
                obs_dict[agent_name] = np.zeros(obs_dim, dtype=np.float32)
                action_dict[agent_name] = 0
                reward_dict[agent_name] = 0.0
                log_prob_dict[agent_name] = 0.0

        # Criticの価値推定（グローバル状態から）
        if 'global_state_tensor' in locals():
            value = mappo.get_value(global_state_tensor)
            # get_valueはすでに.item()でスカラーを返しているので、そのまま代入
            value_cpu = value
        else:
            value_cpu = 0.0

        # バッファに保存（valueはCPU上のスカラー値）
        mappo.buffer.store(
            obs_dict, action_dict, reward_dict, global_state,
            log_prob_dict, value_cpu, terminated, truncated
        )

        if terminated or truncated:
            break

    # エピソード終了後の処理
    reward_buffer.append(episode_reward)
    capture_buffer.append(episode_captures)

    avg_reward = np.mean(reward_buffer)
    avg_captures = np.mean(capture_buffer)

    print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
          f"Captures = {episode_captures}, "
          f"Avg Reward = {avg_reward:.2f}, "
          f"Avg Captures = {avg_captures:.2f}")

    # Advantage計算
    mappo.buffer.compute_advantages(gamma=gamma, gae_lambda=gae_lambda)

    # PPO更新（複数エポック）→ 損失・エントロピーを記録
    actor_losses = []
    critic_losses = []
    entropies = []

    for _ in range(update_epochs):
        batch = mappo.buffer.sample(batch_size)
        # MAPPO.update が (actor_loss, critic_loss, entropy) を返す前提
        actor_loss, critic_loss, entropy = mappo.update(batch)

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        entropies.append(entropy)

    avg_actor_loss = np.mean(actor_losses)
    avg_critic_loss = np.mean(critic_losses)
    avg_entropy = np.mean(entropies)

    print(f"Actor Loss: {avg_actor_loss:.4f} | "
          f"Critic Loss: {avg_critic_loss:.4f} | "
          f"Avg Entropy: {avg_entropy:.4f}")

    # バッファをクリア（次のエピソード用）
    mappo.buffer.clear()

    # チェックポイント保存（特定のインターバルで）
    if episode % checkpoint_interval == 0:
        save_checkpoint(mappo, episode, checkpoint_dir=CHECKPOINT_DIR, device=device)

env.close()
print("=== 学習終了 ===")