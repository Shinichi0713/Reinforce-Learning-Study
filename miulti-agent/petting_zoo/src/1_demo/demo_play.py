# 2. インポート
import numpy as np
import imageio
from pettingzoo.mpe import simple_spread_v3

# 3. 環境作成（render_mode を rgb_array に設定）
env = simple_spread_v3.env(
    N=3,
    local_ratio=0.5,
    max_cycles=100,
    render_mode="rgb_array"  # 画像配列を取得するモード
)

# 4. 録画用のフレームリスト
frames = []

# 5. 1エピソード分を実行し、各ステップの画像を保存
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

    # 現在のフレームを取得してリストに追加
    frame = env.render()
    frames.append(frame)

env.close()

# 6. フレームを MP4 動画として保存
imageio.mimsave("episode.mp4", frames, fps=10)

print("動画を episode.mp4 として保存しました。")