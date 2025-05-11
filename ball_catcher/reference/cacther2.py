from ple import PLE
from ple.games.catcher import Catcher
import time

game = Catcher(width=256, height=256)
env = PLE(game, fps=30, display_screen=True)
env.init()

for episode in range(3):
    env.reset_game()
    while not env.game_over():
        action = env.getActionSet()[0]  # 例: 左に動かす
        env.act(action)
        time.sleep(0.02)
env.close()
