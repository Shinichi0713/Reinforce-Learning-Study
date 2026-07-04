import os
import numpy as np
import torch
import imageio
from PIL import Image

# =====================================================================
# 設定（学習コードと共通）
# =====================================================================
max_cycles = 500
CHECKPOINT_DIR = "/content/drive/MyDrive/rl_pursuit"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🌟 行動履歴付き（obs_dim=236）に対応したラッパーで初期化（render_mode="human"）
env_wrapper = PursuitWrapper(render_mode="human", max_cycles=max_cycles)
num_agents = env_wrapper.num_agents
obs_dim = env_wrapper.obs_dim          # 🌟 自動的に 236 (196 + 40) になります
state_dim = env_wrapper.state_dim      # 🌟 自動的に 1888 (236 x 8) になります
action_dim = env_wrapper.action_dim

# MAPPO の初期化 (内部アーキテクチャはTransformer)
mappo = MAPPO(
    num_agents=num_agents,
    obs_dim=obs_dim,
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=0.99,
    gae_lambda=0.95,
    device=device
)

# 🛠️ チェックポイント復元関数の安全ラップ（学習コードと統一）
def load_checkpoint_safely(model_obj, path, device):
    try:
        checkpoint = torch.load(path, map_location=device)
        model_obj.actor.load_state_dict(checkpoint['actor_state_dict'])
        model_obj.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"💾 チェックポイントを読み込みました (Episode {checkpoint['episode']})")
        return checkpoint['episode']
    except Exception as e:
        print(f"エラー: チェックポイントの読み込み中にエラーが発生しました: {e}")
        exit(1)

path_checkpoint = os.path.join(CHECKPOINT_DIR, "mappo_episode.pth")
if os.path.exists(path_checkpoint):
    load_checkpoint_safely(mappo, path_checkpoint, device=device)
else:
    print("エラー: チェックポイントが見つかりません。")
    exit(1)

# ネットワークを評価モードに
mappo.actor.eval()
mappo.critic.eval()

print("=== MAPPO 推論モード（Transformer & gif保存付き）開始 ===")
print(f"エージェント数: {num_agents}")
print(f"観測次元 (空間196 + 行動履歴40): {obs_dim}")
print(f"使用デバイス: {device}")

# =====================================================================
# フレーム保存用の設定
# =====================================================================
video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)
frames = []  # 各ステップの画像を保存するリスト

# =====================================================================
# 推論ループ（1エピソード分）
# =====================================================================
env_wrapper.reset()
episode_reward = 0.0
episode_captures = 0

# AEC環境のループ
for agent in env_wrapper.env.agent_iter():
    # 現在アクティブなエージェントの終了フラグを取得
    _, _, terminated, truncated, _ = env_wrapper.env.last(agent)

    if terminated or truncated:
        action = None
    else:
        # 1. 全エージェントの最新236次元観測（行動履歴含む）を集約
        obs_list = []
        for i in range(num_agents):
            agent_name = f'pursuer_{i}'
            if agent_name in env_wrapper.env.agents:
                agent_obs = env_wrapper.get_obs(agent_name)
                obs_list.append(agent_obs if agent_obs is not None else np.zeros(obs_dim, dtype=np.float32))
            else:
                obs_list.append(np.zeros(obs_dim, dtype=np.float32))

        # 2. テンソル化して Transformer Actor に入力
        # 🌟 viewエラー対策として .contiguous() を追加
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device).contiguous()

        with torch.no_grad():
            actions_np, _ = mappo.get_action(obs_tensor)

        # 現在のエージェントに対応する行動を取得
        agent_idx = int(agent.split('_')[-1])
        action = actions_np[agent_idx]

    # 3. 環境を1ステップ進める
    obs, reward, terminated, truncated, info = env_wrapper.step(agent, action)
    episode_reward += reward

    # 🌟 4. 【最適化】1サイクル（全員分の行動が1周）終わったタイミングで画面をキャプチャ
    if agent == f'pursuer_{num_agents - 1}':
        frame = env_wrapper.render()
        if frame is not None:
            frames.append(frame)

    # 🌟 5. 捕獲成功判定
    if terminated and not truncated:
        # 終了ステップの最後のフレームも確実に記録
        frame = env_wrapper.render()
        if frame is not None:
            frames.append(frame)
        episode_captures += 1
        break

    if truncated:
        break

print(f"推論結果: Total Hybrid Reward = {episode_reward:.2f}, Captures = {episode_captures}")

env_wrapper.env.close()
print("=== MAPPO 推論モード（フレーム収集）終了 ===")

# =====================================================================
# 収集したフレームを gif に保存
# =====================================================================
if len(frames) > 0:
    gif_path = os.path.join(video_folder, "pursuit_mappo.gif")
    print(f"gif 保存中: {gif_path} (総フレーム数: {len(frames)})")

    # フレームを gif に書き出し
    with imageio.get_writer(gif_path, fps=12) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"gif 保存完了: {gif_path}")
else:
    print("⚠️ フレームが収集されていません。render() の戻り値を確認してください。")

print("=== gif 保存処理完了 ===")