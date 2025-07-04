import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import deque

def train(env, agent, n_episodes=8000, model_type='unk', env_type='unk', score_limit=300.0, explore_episode=50, test_f=200, max_t_step=750):
    scores_deque = deque(maxlen=100)
    scores = []
    test_scores = []
    max_score = -np.inf
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        done = False
        agent.train_mode()
        t = int(0)
        while not done and t < max_t_step:    
            t += int(1)
            action = agent.get_action(state, explore=True)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            agent.memory.add(state, action, reward, next_state, info["dead"])

            state = next_state
            score += reward
            agent.step_end()
            #if i_episode>explore_episode:
            #    env.render()

        if i_episode>explore_episode:
            agent.episode_end()
            for i in range(t):
                agent.learn_one_step()

        scores_deque.append(score)
        avg_score_100 = np.mean(scores_deque)
        scores.append((i_episode, score, avg_score_100))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, avg_score_100, score), end="")

        if i_episode % test_f == 0 or avg_score_100>score_limit:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.eval_mode() # test in eval mode.
            test_score = test(env, agent, render=False, n_times=20)
            test_scores.append((i_episode, test_score))
            agent.save_ckpt(model_type, env_type,'ep'+str(int(i_episode)))
            if avg_score_100>score_limit:
                break
            agent.train_mode() # when the test done, come back to train mode.

    return np.array(scores).transpose(), np.array(test_scores).transpose()

def test(env, agent, render=True, max_t_step=1000, explore=False, n_times=1):
    sum_scores = 0
    for i in range(n_times):
        state = env.reset()
        score = 0
        done=False
        t = int(0)
        while not done and t < max_t_step:
            t += int(1)
            action = agent.get_action(state, explore=explore)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            #print(action)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            state = next_state
            score += reward
            if render:
                env.render()
        sum_scores += score
    mean_score = sum_scores/n_times
    print('\rTest Episodes\tMean Score: {:.2f}'.format(mean_score))
    return mean_score

