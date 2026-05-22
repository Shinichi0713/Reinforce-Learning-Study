import glob
import io
import base64
import cv2
import numpy as np
import gymnasium as gym
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from pettingzoo.atari import wizard_of_wor_v3

# 5. Colab上で再生
def show_local_video(path):
    video_file = io.open(path, 'r+b').read()
    encoded = base64.b64encode(video_file)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))

# 1. 仮想ディスプレイの起動
display = Display(visible=0, size=(1400, 900))
display.start()

# 2. 環境の構築
env = wizard_of_wor_v3.env(render_mode="rgb_array")
env.reset()

frames = []

# --- MARL用の設定 ---
# 各エージェントの直前の状態を保持する辞書
prev_data = {agent: {"obs": None, "action": None} for agent in env.possible_agents}
experience_buffer = {agent: [] for agent in env.possible_agents}

def policy(agent, observation):
    # ここに将来的にモデル（Q-Network等）を組み込む
    return env.action_space(agent).sample()

# 3. 実行ループ
for agent in env.agent_iter():
    # 現在のターンのエージェントの情報を取得
    obs, reward, termination, truncation, info = env.last()
    done = termination or truncation

    # 【重要】前回の自分の行動の結果（報酬と次状態）をバッファに記録
    if prev_data[agent]["action"] is not None:
        experience_buffer[agent].append((
            prev_data[agent]["obs"],
            prev_data[agent]["action"],
            reward,
            obs,
            done
        ))

    if done:
        action = None
    else:
        # 現在の観測に基づいて行動を選択
        action = policy(agent, obs)
        # 次の自分のターンのために現在の情報を保存
        prev_data[agent]["obs"] = obs
        prev_data[agent]["action"] = action

    # 行動を実行
    env.step(action)
    
    # 画面キャプチャ（全プレイヤー共通）
    frames.append(env.render())
    
    if len(frames) > 1000:
        break

env.close()

# 4. 動画の書き出し
video_path = 'wizard_marl_fixed.mp4'
height, width, _ = frames[0].shape
video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
for f in frames:
    video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
video.release()

show_local_video(video_path) # 前述の関数を使用