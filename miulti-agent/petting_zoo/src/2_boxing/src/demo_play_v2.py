import os
import torch
import numpy as np
import imageio
from google.colab import drive
import supersuit as ss
from pettingzoo.atari import boxing_v2

# 1. 環境構築
def get_eval_env(render_mode="rgb_array"):
    env = boxing_v2.parallel_env(render_mode=render_mode)
    env = ss.resize_v1(env, 84, 84)
    env = ss.color_reduction_v0(env, mode='full')
    env = ss.dtype_v0(env, "float32")
    env = ss.frame_stack_v1(env, 4)
    return env

# 2. パス・デバイス設定
drive.mount('/content/drive')
VIDEO_DIR = "/content/drive/MyDrive/rl_boxing/videos"
CHECKPOINT_DIR = "/content/drive/MyDrive/rl_boxing/checkpoints"
os.makedirs(VIDEO_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 学習済みモデルのロード
# ※ MAPPOAgentクラスが定義されている必要があります
agent = MAPPOAgent(action_space_n=18).to(device)

def find_latest_checkpoint(checkpoint_dir):
    import glob
    import re
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "mappo_agent_iter_*.pth"))
    if not checkpoints: return None
    ids = [int(re.findall(r"iter_(\d+)", f)[0]) for f in checkpoints]
    return checkpoints[np.argmax(ids)]

checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)

if checkpoint_path:
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    print("Checkpoint loaded.")
else:
    print("Checkpoint not found."); exit()

# 4. 実行と録画
env = get_eval_env(render_mode="rgb_array")
obs_dict, infos = env.reset()

frames = []
total_reward_1p = 0.0
total_reward_2p = 0.0
max_steps = 5000 # 十分な長さを確保

print("Recording in progress: Trained(1P) vs Random(2P)")

for s in range(max_steps):
    # 画面キャプチャ（1P視点のみでなく全画面）
    frame = env.render()
    frames.append(frame)

    actions = {}
    
    # --- 1P: 学習済みエージェントの推論 ---
    if 'first_0' in obs_dict:
        obs = obs_dict['first_0']
        # 軸入れ替え (H, W, C) -> (C, H, W)
        obs_transposed = np.transpose(obs, (2, 0, 1)) 
        # テンソル化 & 正規化
        obs_tensor = torch.from_numpy(obs_transposed).to(device).unsqueeze(0)
        obs_tensor = obs_tensor / 255.0
        
        with torch.no_grad():
            action, _, _ = agent.get_action(obs_tensor)
            actions['first_0'] = action.item()

    # --- 2P: ランダムアクション ---
    if 'second_0' in obs_dict:
        # action_space からランダムにサンプリング
        actions['second_0'] = env.action_space('second_0').sample()

    # --- ステップ実行 (ここを1回にする) ---
    next_obs_dict, rewards, terms, truncs, infos = env.step(actions)
    
    # 報酬の加算
    total_reward_1p += rewards.get('first_0', 0)
    total_reward_2p += rewards.get('second_0', 0)
    
    obs_dict = next_obs_dict

    # 終了判定
    if any(terms.values()) or any(truncs.values()):
        print(f"Game Over at step {s}")
        break

env.close()

# 5. 動画の書き出し
video_filename = f"boxing_vs_random_reward_{int(total_reward_1p)}.mp4"
video_path = os.path.join(VIDEO_DIR, video_filename)
imageio.mimsave(video_path, frames, fps=30)

print("-" * 30)
print(f"Finished! Total Steps: {len(frames)}")
print(f"1P (Trained) Total Reward: {total_reward_1p}")
print(f"2P (Random)  Total Reward: {total_reward_2p}")
print(f"Video saved to: {video_path}")