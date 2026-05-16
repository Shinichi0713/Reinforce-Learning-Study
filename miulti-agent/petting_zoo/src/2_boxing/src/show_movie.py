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

# 2. パス設定
drive.mount('/content/drive')
VIDEO_DIR = "/content/drive/MyDrive/rl_boxing/videos"
CHECKPOINT_DIR = "/content/drive/MyDrive/rl_boxing/checkpoints"
os.makedirs(VIDEO_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. モデルのロード
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
total_reward = 0.0
max_steps = 2000

print("Recording in progress...")

for s in range(max_steps):
    # レンダリング画像の取得
    frame = env.render()
    frames.append(frame)

    actions = {}
    with torch.no_grad():
        for agent_id, obs in obs_dict.items():
            # --- ここを修正： (84, 84, 4) -> (4, 84, 84) に変換 ---
            # numpy配列の軸を入れ替える
            obs_transposed = np.transpose(obs, (2, 0, 1)) 
            
            # テンソル化してバッチ次元を追加 (1, 4, 84, 84)
            obs_tensor = torch.from_numpy(obs_transposed).to(device).unsqueeze(0)
            
            # 推論
            action, _, _ = agent.get_action(obs_tensor)
            actions[agent_id] = action.item()

    # ステップ実行
    next_obs_dict, rewards, terms, truncs, infos = env.step(actions)
    
    # 1P(first_0)側の報酬を加算
    total_reward += rewards.get('first_0', 0)
    obs_dict = next_obs_dict

    if any(terms.values()) or any(truncs.values()):
        break

env.close()

# 5. 動画の書き出し
video_filename = f"boxing_eval_iter780_reward_{int(total_reward)}.mp4"
video_path = os.path.join(VIDEO_DIR, video_filename)
imageio.mimsave(video_path, frames, fps=30)

print("-" * 30)
print(f"Finished! Total Steps: {len(frames)}")
print(f"Total Reward (1P): {total_reward}")
print(f"Video saved to: {video_path}")