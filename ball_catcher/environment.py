
import sys, os
dir_root = '/'.join(os.path.dirname(os.path.abspath(__file__)).replace("\\", '/').split("/")[:-1])
print(dir_root)
sys.path.append(dir_root + '/ple-cited')
from ple.ple import PLE
from ple.games.catcher import Catcher
import pygame
pygame.init()


class BallCatcherEnv:
    def __init__(self, width=256, height=256, fps=240, display_screen=True):
        self.game = Catcher(width=width, height=height)
        self.env = PLE(self.game, fps=fps, display_screen=display_screen)
        self.env.init()
        self.actions = self.env.getActionSet()  # [左, 右, 何もしない]

    def reset(self):
        return self.env.reset_game()

    def step(self, action):
        reward = self.env.act(action)
        done = self.env.game_over()
        return reward, done

    def get_keyevent(self):
        action_user = self.actions[2]
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action_user = self.actions[0]
                elif event.key == pygame.K_RIGHT:
                    action_user = self.actions[1]
                elif event.key == pygame.K_ESCAPE:
                    action_user = None
        return action_user

    def close(self):
        self.env.close()

    def get_state(self):
        return self.game.getGameState()
    def get_action_set(self):
        return self.env.getActionSet()
    def get_action_space(self):
        return len(self.actions)
    def get_observation_space(self):
        return self.game.getGameState()
    def get_state_size(self):
        return len(self.game.getGameState())
    def get_action_size(self):
        return len(self.actions)
    def get_fps(self):
        return self.env.fps
    def get_width(self):
        return self.game.width
    def get_height(self):
        return self.game.height
    def get_display_screen(self):
        return self.env.display_screen
    def set_display_screen(self, display_screen):
        self.env.display_screen = display_screen
    def get_game(self):
        return self.game
    

if __name__ == "__main__":
    import time
    # 環境の作成
    env = BallCatcherEnv()
    state = env.reset()
    done = False
    for i in range(3):
        state = env.reset()
        while not done:
            # action = env.get_action_set()[2]  # 例: 左に動かす
            action = env.get_keyevent()
            reward, done = env.step(action)
            time.sleep(0.02)
    print("Game Over")