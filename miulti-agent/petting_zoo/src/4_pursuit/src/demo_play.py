import os
import numpy as np
import torch
import imageio
from PIL import Image

# =====================================================================
# 設定（学習コードと共通）
# =====================================================================
max_cycles = 500
CHECKPOINT_DIR = "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PursuitWrapper の初期化（render_mode="human" で可視化）
env_wrapper = PursuitWrapper(render_mode="human", max_cycles=max_cycles)
num_agents = env_wrapper.num_agents
obs_dim = env_wrapper.obs_dim
state_dim = env_wrapper.state_dim
action_dim = env_wrapper.action_dim

# MAPPO の初期化（学習時と同じ構成）
mappo = MAPPO(
    num_agents=num_agents,
    obs_dim=obs_dim,
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=0.99,
    gae_lambda=0.95,
    device=device
)

# チェックポイントの読み込み
path_checkpoint = os.path.join(CHECKPOINT_DIR, "mappo_episode.pth")
if os.path.exists(path_checkpoint):
    load_checkpoint(mappo, path_checkpoint, device=device)
    print("チェックポイントを読み込みました。")
else:
    print("エラー: チェックポイントが見つかりません。")
    exit(1)

# ネットワークを評価モードに
mappo.actor.eval()
mappo.critic.eval()

print("=== MAPPO 推論モード（gif保存付き）開始 ===")
print(f"エージェント数: {num_agents}")
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

# AEC環境のループ（PursuitWrapper の env を使う）
for agent in env_wrapper.env.agent_iter():
    # 現在アクティブなエージェントの終了フラグを取得
    _, _, terminated, truncated, _ = env_wrapper.env.last(agent)
    
    if terminated or truncated:
        action = None
    else:
        # 1. 全エージェントの最新観測を集約
        obs_list = []
        for i in range(num_agents):
            agent_name = f'pursuer_{i}'
            agent_obs = env_wrapper.get_obs(agent_name)
            obs_list.append(agent_obs if agent_obs is not None else np.zeros(obs_dim, dtype=np.float32))

        # 2. テンソル化して Actor に入力（推論モード）
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)
        
        with torch.no_grad():
            actions_np, _ = mappo.get_action(obs_tensor)

        # 現在のエージェントに対応する行動を取得
        agent_idx = int(agent.split('_')[-1])
        action = actions_np[agent_idx]

    # 3. 環境を1ステップ進める（PursuitWrapper.step を使う）
    reward, terminated, truncated, info = env_wrapper.step(agent, action)
    episode_reward += reward

    # 捕獲報酬をカウント
    if reward == 5.0:
        episode_captures += 1

    # 4. 現在の画面を画像としてキャプチャして保存
    #    ※ render_mode="human" の場合、env_wrapper.env.render() で画面が更新される
    #    ここでは env_wrapper.env の render 結果を取得する方法が環境依存のため、
    #    必要に応じて env_wrapper.env.render() を呼び出し、その戻り値や画面バッファから画像を取得してください。
    #    例: frame = env_wrapper.env.render()  # 戻り値が numpy array の場合
    #        frames.append(frame)
    #
    # もし render() が None を返す場合は、別途 pygame や matplotlib で描画した画面を
    # PIL.Image や numpy array に変換して frames に追加してください。
    #
    # 以下は仮のコード例です（実際の環境に合わせて修正してください）：
    #
    # frame = env_wrapper.env.render()
    # if frame is not None:
    #     frames.append(frame)

    if terminated or truncated:
        break

print(f"推論結果: Reward = {episode_reward:.2f}, Captures = {episode_captures}")

env_wrapper.env.close()
print("=== MAPPO 推論モード（フレーム収集）終了 ===")

# =====================================================================
# 収集したフレームを gif に保存
# =====================================================================
if len(frames) > 0:
    gif_path = os.path.join(video_folder, "pursuit_mappo.gif")
    print(f"gif 保存中: {gif_path}")

    # フレームを gif に書き出し（fps はお好みで調整）
    with imageio.get_writer(gif_path, fps=10) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"gif 保存完了: {gif_path}")
else:
    print("フレームが収集されていません。render() の実装を確認してください。")

print("=== gif 保存処理完了 ===")