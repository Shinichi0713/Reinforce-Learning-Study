
def evaluate_mappo_cnn(env, mappo, num_agents, obs_dim, state_dim, max_cycles=100):
    """評価用の関数（貪欲方策で1エピソード実行）"""
    env_eval = PursuitWrapper(render_mode="rgb_array", max_cycles=max_cycles)
    env_eval.reset()

    total_reward = 0.0
    frames = []

    for agent in env_eval.env.agent_iter():
        obs = env_eval.get_obs(agent)
        global_state = env_eval.get_global_state()

        if agent not in env_eval.env.agents:
            action = None
            reward = 0.0
            terminated = True
            truncated = True
        else:
            _, _, terminated, truncated, _ = env_eval.env.last(agent, )
            if terminated or truncated:
                action = None
            else:
                # 各エージェントの観測をテンソルにまとめる
                obs_list = []
                for i in range(num_agents):
                    agent_name = f'pursuer_{i}'
                    if agent_name in env_eval.env.agents:
                        agent_obs = env_eval.get_obs(agent_name)
                        if agent_obs is not None:
                            obs_list.append(agent_obs)
                    else:
                        obs_list.append(np.zeros(obs_dim, dtype=np.float32))

                obs_tensor = torch.FloatTensor(np.array(obs_list))
                global_state_tensor = torch.FloatTensor(global_state)

                # 貪欲方策で行動選択
                actions_np, _ = mappo.get_action(obs_tensor, greedy=True)
                agent_idx = int(agent.split('_')[-1])
                action = actions_np[agent_idx]

            reward, terminated, truncated, info = env_eval.step(agent, action)
            total_reward += reward

        # 描画（オプション）
        frame = env_eval.env.render()
        frames.append(frame)

        if terminated or truncated:
            break

    env_eval.close()
    print(f"評価: Reward = {total_reward:.2f}")

    # 動画保存（オプション）
    # save_video(frames, f"evaluation_episode.mp4")

    return total_reward, frames

