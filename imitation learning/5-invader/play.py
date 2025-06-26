from imitation.data.types import Trajectory
from stable_baselines3.common.atari_wrappers import *
import gym
import pyglet
from pyglet.window import key
import time
import pickle

def get_key_state(win, key_handler):
    key_state = set()
    win.dispatch_events()
    for key_code, pressed in key_handler.items():
        if pressed:
            key_state.add(key_code)
    return key_state

def human_expert(_state, win, key_handler):
    key_state = get_key_state(win, key_handler)
    action = 0
    if key.SPACE in key_state:
        action = 1
    elif key.LEFT in key_state:
        action = 3
    elif key.RIGHT in key_state:
        action = 4
    time.sleep(1.0 / 30.0)
    return action

def main():
    record_episodes = 1
    ENV_ID = 'SpaceInvaders-v0'
    env = gym.make(ENV_ID)
    env.render()

    win = pyglet.window.Window(width=300, height=100, vsync=False)
    key_handler = pyglet.window.key.KeyStateHandler()
    win.push_handlers(key_handler)
    pyglet.app.platform_event_loop.start()
    while len(get_key_state(win, key_handler)) == 0:
        time.sleep(1.0 / 30.0)
    
    trajectorys = []
    for i in range(0, record_episodes):
        state = env.reset()
        actions = []
        infos = []
        observations = [state]
        while True:
            env.render()
            action = human_expert(state, win, key_handler)
            state, reward, done, info = env.step(action)
            actions.append(action)
            observations.append(state)
            infos.append(info)
            if done:
                ts = Trajectory(obs=np.array(observations), acts=np.array(actions), infos=np.array(infos))
                trajectorys.append(ts)
                break
    with open("invader_expert.pickle", mode="wb") as f:
        pickle.dump(trajectorys, f)
if __name__ == '__main__':
    main()