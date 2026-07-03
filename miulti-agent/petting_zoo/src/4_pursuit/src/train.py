import os
from collections import deque
import numpy as np
import torch

# =====================================================================
# ハイパーパラメータ & 設定
# =====================================================================
max_episodes = 1000
max_cycles = 500
batch_size = 64
update_epochs = 3
gamma = 0.99
gae_lambda = 0.95
checkpoint_interval = 10
CHECKPOINT_DIR = "/content/drive/MyDrive/rl_pursuit"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🌟 4チャンネル化（obs_dim=196）に対応したラッパーで初期化
env = PursuitWrapper(render_mode=None, max_cycles=max_cycles)
num_agents = env.num_agents
obs_dim = env.obs_dim          # 🌟 自動的に 196 (7x7x4) になります
state_dim = env.state_dim      # 🌟 自動的に 1568 (196x8) になります
action_dim = env.action_dim

# MAPPOの初期化 (内部のActor/CriticがTransformerへと自動構築されます)
mappo = MAPPO(
    num_agents=num_agents,
    obs_dim=obs_dim,
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=gamma,
    gae_lambda=gae_lambda,
    device=device
)

# 🌟 外部にMultiAgentBufferを明示的に紐付け
from __main__ import MultiAgentBuffer # 別ファイル定義の場合は適宜importを修正してください
mappo.buffer = MultiAgentBuffer(num_agents, obs_dim, state_dim, action_dim)

print("=== MAPPO（Transformer & IDベース）学習開始 ===")
print(f"エージェント数: {num_agents}")
print(f"観測次元 (4Chフラット): {obs_dim}")
print(f"グローバル状態次元: {state_dim}")
print(f"行動空間サイズ: {action_dim}")
print(f"使用デバイス: {device}")

# 🛠️ チェックポイント復元関数のバグ防止用安全ラップ
def load_checkpoint_safely(model_obj, path, device):
    try:
        checkpoint = torch.load(path, map_location=device)
        model_obj.actor.load_state_dict(checkpoint['actor_state_dict'])
        model_obj.critic.load_state_dict(checkpoint['critic_state_dict'])
        model_obj.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        model_obj.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        return checkpoint['episode']
    except Exception as e:
        print(f"警告: チェックポイントの読み込み中にエラーが発生しました（アーキテクチャ不一致の可能性）: {e}")
        return 0

# 🛠️ チェックポイント保存関数の安全ラップ
def save_checkpoint_safely(model_obj, ep, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "mappo_episode.pth")
    torch.save({
        'episode': ep,
        'actor_state_dict': model_obj.actor.state_dict(),
        'critic_state_dict': model_obj.critic.state_dict(),
        'optimizer_actor_state_dict': model_obj.optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': model_obj.optimizer_critic.state_dict(),
    }, save_path)
    print(f"💾 チェックポイントを保存しました: {save_path} (Episode {ep})")

path_checkpoint = os.path.join(CHECKPOINT_DIR, "mappo_episode.pth")
start_episode = 0
if os.path.exists(path_checkpoint):
    start_episode = load_checkpoint_safely(mappo, path_checkpoint, device=device)
    print(f"再開エピソード: {start_episode}")

# モニタリング用のバッファ
reward_buffer = deque(maxlen=100)
capture_buffer = deque(maxlen=100)

# =====================================================================
# メイン学習ループ
# =====================================================================
for episode in range(start_episode, max_episodes):
    env.reset()
    episode_reward = 0.0
    episode_captures = 0

    # AEC環境のループ
    for agent in env.env.agent_iter():
        _, _, terminated, truncated, _ = env.env.last(agent)

        if terminated or truncated:
            action = None
        else:
            # 1. 全エージェントの現在の最新4チャンネル観測(196次元)を集約
            obs_list = []
            for i in range(num_agents):
                agent_name = f'pursuer_{i}'
                if agent_name in env.env.agents:
                    agent_obs = env.get_obs(agent_name)
                    obs_list.append(agent_obs if agent_obs is not None else np.zeros(obs_dim, dtype=np.float32))
                else:
                    obs_list.append(np.zeros(obs_dim, dtype=np.float32))

            # 2. テンソル化してポリシー(Transformer Actor)に入力
            obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)  # (num_agents, 196)

            # MAPPOで全エージェント分の行動と確率を同時に推論
            actions_np, log_probs_np = mappo.get_action(obs_tensor)

            # 現在のループ対象エージェントの行動を抽出
            agent_idx = int(agent.split('_')[-1])
            action = actions_np[agent_idx]

        # 3. 環境を1ステップ進める obs, hybrid_reward, terminated, truncated, info
        obs, reward, terminated, truncated, info, count_capture = env.step(agent, action)
        episode_reward += reward

        # Pursuit環境特有の捕獲報酬(+5)をカウント
        episode_captures += count_capture if count_capture else 0

        # 4. バッファに保存するためのステップ全体の辞書データを構築
        obs_dict = {}
        action_dict = {}
        reward_dict = {}
        log_prob_dict = {}
        value_dict = {}

        # 全エージェントの情報を辞書にマッピング
        for i in range(num_agents):
            agent_name = f'pursuer_{i}'
            if agent_name in env.env.agents:
                agent_obs = env.get_obs(agent_name)
                obs_dict[agent_name] = agent_obs if agent_obs is not None else np.zeros(obs_dim, dtype=np.float32)

                # 現在の行動決定エージェント以外は0（プレースホルダー）
                if agent_name == agent:
                    action_dict[agent_name] = action if action is not None else 0
                    reward_dict[agent_name] = reward
                else:
                    action_dict[agent_name] = 0
                    reward_dict[agent_name] = 0.0

                log_prob_dict[agent_name] = log_probs_np[i] if 'log_probs_np' in locals() else 0.0
            else:
                obs_dict[agent_name] = np.zeros(obs_dim, dtype=np.float32)
                action_dict[agent_name] = 0
                reward_dict[agent_name] = 0.0
                log_prob_dict[agent_name] = 0.0

        # 5. Centralized Transformer Criticによる価値推定
        global_state_list = [obs_dict[f'pursuer_{k}'] for k in range(num_agents)]
        # 🌟 (1, num_agents, 196) の形状で綺麗にテンソル化
        global_state_tensor = torch.FloatTensor(np.array(global_state_list)).unsqueeze(0).to(device)

        for i in range(num_agents):
            agent_name = f'pursuer_{i}'
            if agent_name in env.env.agents:
                target_id_tensor = torch.tensor([i], dtype=torch.long, device=device)
                with torch.no_grad():
                    # 🌟 Transformer Criticを介して V(s) を抽出
                    val = mappo.critic(global_state_tensor, target_id_tensor).item()
                value_dict[agent_name] = val
            else:
                value_dict[agent_name] = 0.0

        # 6. 1次元化した global_state と共に、マルチエージェントバッファへ格納
        global_state_flat = np.array(global_state_list).flatten() # (num_agents * 196 = 1568,)
        mappo.buffer.store(
            obs_dict, action_dict, reward_dict, global_state_flat,
            log_prob_dict, value_dict, terminated, truncated
        )
        # print(value_dict)
        if terminated or truncated:
            break

    # =====================================================================
    # エピソード終了後のネットワーク更新処理
    # =====================================================================
    reward_buffer.append(episode_reward)
    capture_buffer.append(episode_captures)

    avg_reward = np.mean(reward_buffer)
    avg_captures = np.mean(capture_buffer)

    print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
          f"Captures = {episode_captures}, "
          f"Avg Reward = {avg_reward:.2f}, "
          f"Avg Captures = {avg_captures:.2f}")

    # アドバンテージ（GAE）とリターンの計算
    mappo.buffer.compute_advantages(gamma=gamma, gae_lambda=gae_lambda)

    # PPOの更新（ミニバッチサンプリング）
    actor_losses, critic_losses, entropies = [], [], []

    for _ in range(update_epochs):
        batch = mappo.buffer.sample(batch_size)
        if batch is not None:
            # 🌟 エポック数を内部でループさせず、サンプリングごとに1回更新を実行
            actor_loss, critic_loss, entropy = mappo.update(batch, epochs=1)

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropies.append(entropy)

    if len(actor_losses) > 0:
        print(f"Actor Loss: {np.mean(actor_losses):.4f} | "
              f"Critic Loss: {np.mean(critic_losses):.4f} | "
              f"Avg Entropy: {np.mean(entropies):.4f}")

    # 次のエピソードに備えてバッファをリセット
    mappo.buffer.clear()

    # 一定間隔でチェックポイントをセーブ
    if episode % checkpoint_interval == 0 and episode > 0:
        save_checkpoint_safely(mappo, episode, checkpoint_dir=CHECKPOINT_DIR)

env.close()
print("=== MAPPO（Transformer & IDベース）学習完了 ===")