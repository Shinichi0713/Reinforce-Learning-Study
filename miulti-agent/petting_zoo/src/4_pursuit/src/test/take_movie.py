import imageio
from pettingzoo.sisl import pursuit_v4

# GIFの保存先ファイル名
video_filename = "pursuit_random_agents.gif"

# 環境生成（render_mode="rgb_array" で画像取得可能にする）
env = pursuit_v4.env(render_mode="rgb_array")

# フレームを格納するリスト
frames = []

# 環境リセット
env.reset()

max_frames = 400  # 保存したい最大フレーム数
frame_count = 0

# ランダムエージェントで実行し、各ステップの画面を保存
for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()

    # 画面を画像として取得
    frame = env.render()
    frames.append(frame)
    frame_count += 1

    # 指定フレーム数に達したらループを抜ける
    if frame_count >= max_frames:
        break

    if terminated or truncated:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

env.close()

# フレームをGIFにエンコードして保存
from scipy.ndimage import zoom
from PIL import Image
import numpy as np

def resize_frame(frame, scale=0.5):
    h, w, c = frame.shape
    resized = zoom(frame, (scale, scale, 1), order=1)
    return resized.astype(frame.dtype)

def quantize_frame(frame, colors=64):
    img = Image.fromarray(frame)
    img_quantized = img.quantize(colors=colors)
    return np.array(img_quantized.convert("RGB"))

# 軽量化したGIFを保存
with imageio.get_writer("pursuit_small.gif", fps=5) as video:
    for frame in frames:
        small = resize_frame(frame, scale=0.5)
        q_small = quantize_frame(small, colors=16)
        video.append_data(q_small)

print(f"GIFを保存しました: {video_filename}")