import gymnasium as gym
import ale_py
from gymnasium.wrappers import RecordVideo
import glob
import io
import base64
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display

# 仮想ディスプレイの起動
display = Display(visible=0, size=(1400, 900))
display.start()

def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        # 最新の動画ファイルを取得
        mp4 = max(mp4list, key=lambda x: glob.os.path.getctime(x))
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("動画ファイルが見つかりませんでした。")

# --- メイン実行部分 ---

# 環境の作成 (WizardOfWorを指定)
env = gym.make("ALE/WizardOfWor-v5", render_mode="rgb_array")

# 動画保存の設定
env = RecordVideo(env, video_folder='./video', episode_trigger=lambda episode_id: True)

# リセット
observation, info = env.reset()

# 1000ステップ実行（ランダムアクション）
for _ in range(1000):
    action = env.action_space.sample() 
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()

# 動画の表示
show_video()