import random
import pyglet
import gym
import time
from pyglet.window import key
from stable_baselines.gail import generate_expert_traj

# 環境の生成
env = gym.make('CartPole-v1')
env.reset()
env.render()

# キーイベント用のウィンドウの生成
win = pyglet.window.Window(width=300, height=100, vsync=False)
key_handler = pyglet.window.key.KeyStateHandler()
win.push_handlers(key_handler)
pyglet.app.platform_event_loop.start()

# キー状態の取得
def get_key_state():
   key_state = set()
   win.dispatch_events()
   for key_code, pressed in key_handler.items():
       if pressed:
           key_state.add(key_code)
   return key_state

# キー入力待ち
while len(get_key_state()) == 0:
   time.sleep(1.0/30.0)

# デモの行動の指定
def dummy_expert(_obs):
   # キー状態の取得
   key_state = get_key_state()

   # 行動の選択
   action = 0
   if key.LEFT in key_state:
       action = 0
   elif key.RIGHT in key_state:
       action = 1

   # スリープ
   time.sleep(1.0/2.0)

   # 環境の描画
   env.render()

   # 行動の選択
   return action

# デモの記録
generate_expert_traj(dummy_expert, 'cartpole_traj', env, n_episodes=5)