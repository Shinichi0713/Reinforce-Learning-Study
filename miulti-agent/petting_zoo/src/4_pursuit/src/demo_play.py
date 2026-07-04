import os
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# =====================================================================
# 設定（学習コードと共通）
# =====================================================================
max_cycles = 500
CHECKPOINT_DIR = "/content/drive/MyDrive/rl_pursuit"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🌟 4チャンネル化（obs_dim=196）に対応したラッパーで初期化（render_mode="human"）
env_wrapper = PursuitWrapper(render_mode="human", max_cycles=max_cycles)
num_agents = env_wrapper.num_agents
obs_dim = env_wrapper.obs_dim
state_dim = env_wrapper.state_dim
action_dim = env_wrapper.action_dim

# フォントの設定（環境に応じてパスを調整）
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

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
        # 1. 全エージェントの最新観測（196次元フラット）を集約
        obs_list = []
        for i in range(num_agents):
            agent_name = f'pursuer_{i}'
            if agent_name in env_wrapper.env.agents:
                agent_obs = env_wrapper.get_obs(agent_name)
                obs_list.append(agent_obs if agent_obs is not None else np.zeros(obs_dim, dtype=np.float32))
            else:
                obs_list.append(np.zeros(obs_dim, dtype=np.float32))

        # 2. テンソル化して Transformer Actor に入力（(num_agents, 196) の形状）
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(device)

        with torch.no_grad():
            actions_np, _ = mappo.get_action(obs_tensor)

        # 現在のエージェントに対応する行動を取得
        agent_idx = int(agent.split('_')[-1])
        action = actions_np[agent_idx]

    # 🌟 3. 環境を1ステップ進める（戻り値が5つに変更された点に対応！）
    obs, reward, terminated, truncated, info, count_capture = env_wrapper.step(agent, action)
    episode_captures += count_capture if count_capture else 0
    episode_reward += reward

    # 🌟 4. 【最適化】1サイクル（全員分の行動が1周）終わったタイミングで画面をキャプチャ
    # エージェントごとに毎ステップ保存すると、1マスの微小変化でGIFのコマ数が膨大になりテンポが悪くなるのを防ぎます
    if agent == f'pursuer_{num_agents - 1}':
        frame = env_wrapper.render()
        if frame is not None:
            # NumPy配列をPIL Imageに変換
            pil_img = Image.fromarray(frame)

            # 描画オブジェクトを作成
            draw = ImageDraw.Draw(pil_img)

            # 右下の座標を計算（マージン10px）
            text = f"Captures: {episode_captures}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = pil_img.width - text_width - 10
            y = pil_img.height - text_height - 10

            # テキストを描画（白文字＋黒縁など）
            draw.rectangle([x-2, y-2, x+text_width+2, y+text_height+2], fill="black")
            draw.text((x, y), text, fill="white", font=font)

            # PIL ImageをNumPy配列に戻して保存
            frame_with_text = np.array(pil_img)
            frames.append(frame_with_text)

    # 🌟 5. 捕獲成功判定の修正
    # ハイブリッド報酬化により reward の値が変動するため、
    # ラッパー内部で完全捕獲成功（True Capture）の判定と同期する形で終了フラグをチェックします
    if terminated and not truncated:
        # 終了ステップの最後のフレームも確実に記録
        frame = env_wrapper.render()
        if frame is not None:
            frames.append(frame)
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

    # フレームを gif に書き出し（fps はお好みで調整。10〜15が滑らかで見やすいです）
    with imageio.get_writer(gif_path, fps=12) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"gif 保存完了: {gif_path}")
else:
    print("⚠️ フレームが収集されていません。render() の戻り値を確認してください。")

print("=== gif 保存処理完了 ===")