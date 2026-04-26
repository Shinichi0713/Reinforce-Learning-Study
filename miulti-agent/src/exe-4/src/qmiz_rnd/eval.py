def run_multiple_episodes_and_save_best(agent, env, num_episodes=10, filename="best_episode.mp4", fps=5):
    best_reward = -float("inf")
    best_actions_list = None

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        actions_list = []

        while not done and step_count < env.max_steps:
            actions = agent.act(obs, explore=False)
            actions_list.append(actions)
            next_obs, rewards, done, _ = env.step(actions)
            total_reward += sum(rewards)
            obs = next_obs
            step_count += 1

        print(f"Episode {ep}: total_reward = {total_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_actions_list = actions_list

    # 最良エピソードを動画に保存
    env.reset()
    env.save_render_video(best_actions_list, filename=filename, fps=fps)
    print(f"Best episode (reward={best_reward:.2f}) saved to {filename}")

# 使用例
run_multiple_episodes_and_save_best(agent, env, num_episodes=10, filename="best_qmix_rnd.mp4")