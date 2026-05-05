import os
from pettingzoo.atari import boxing_v2
from PIL import Image
from google.colab import files

# 1. 環境の初期化
# ボクシング環境を呼び出し。render_mode="rgb_array" で画像を取得可能にします。
env = boxing_v2.env(render_mode="rgb_array")
env.reset()

frames = []
max_frames = 300  # GIFの長さを調整（約10〜15秒分）

print("シミュレーション実行中...")

# 2. メインループ（エージェントがランダムに行動）
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        # ランダムに行動を選択
        action = env.action_space(agent).sample()

    env.step(action)
    
    # 画面をキャプチャしてリストに保存
    # env.render() は numpy 配列を返すので、PIL.Image に変換
    display_frame = env.render()
    frames.append(Image.fromarray(display_frame))
    
    if len(frames) >= max_frames:
        break

env.close()

# 3. GIFとして保存
gif_filename = "atari_boxing.gif"
frames[0].save(
    gif_filename,
    save_all=True,
    append_images=frames[1:],
    duration=40,  # 1フレームあたりの時間（ミリ秒）
    loop=0        # ループ回数（0は無限ループ）
)

print(f"保存完了: {gif_filename}")

# 4. ローカルPCにダウンロード
files.download(gif_filename)