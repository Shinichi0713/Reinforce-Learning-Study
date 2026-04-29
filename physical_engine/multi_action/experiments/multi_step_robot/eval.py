import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

def run_demo(env, agent, num_episodes=3, max_steps=200, gif_dir="demos"):
    """
    学習済みエージェントでデモを実行し、gifを保存する
    """
    os.makedirs(gif_dir, exist_ok=True)

    for ep in range(num_episodes):
        print(f"=== Episode {ep} ===")
        obs, info = env.reset()
        env.start_recording()  # 録画開始

        total_reward = 0.0
        step_count = 0

        for step in range(max_steps):
            # エージェントの行動を取得（deterministic=Trueで確定的な行動）
            action = agent.get_action(obs, deterministic=True)

            # 録画しながらステップを進める
            obs, reward, done, truncated, info = env.step_with_record(action)
            total_reward += reward
            step_count += 1

            if done:
                break

        # 録画停止
        env.stop_recording()

        # gifを保存
        gif_path = os.path.join(gif_dir, f"demo_ep{ep}.gif")
        env.save_gif(gif_path, duration=100)

        print(f"Episode {ep}: Reward = {total_reward:.2f}, Steps = {step_count}")
        print(f"GIF saved to {gif_path}")
        print()

run_demo(env, agent, num_episodes=3, gif_dir="demos")