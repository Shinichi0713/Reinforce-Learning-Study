def evaluate_agent(agent, env_wrapper, render=False, max_steps=200, save_video=False, video_filename="agent_episode.mp4", fps=5):
    """
    学習済みエージェントを1エピソード評価し、必要に応じて動画を保存する

    Parameters
    ----------
    agent : MAZeroAgent
        学習済みエージェント
    env_wrapper : EnvWrapper
        環境ラッパー
    render : bool
        Trueなら可視化（DroneDeliveryEnv.render()を呼ぶ）
    max_steps : int
        最大ステップ数
    save_video : bool
        Trueならエピソードを動画として保存
    video_filename : str
        保存する動画ファイル名
    fps : int
        動画のフレームレート
    """
    # モデルを評価モードに
    agent.model.eval()

    state = env_wrapper.reset()
    total_rewards = [0, 0]
    step = 0
    done = False

    # 動画保存用に行動リストを記録
    agent_actions_list = []

    if render:
        env_wrapper.env.render()

    while not done and step < max_steps:
        # 行動選択（探索なし、決定論的）
        with torch.no_grad():
            actions = agent.select_action(state)

        # 行動を記録
        agent_actions_list.append(actions)

        next_state, rewards, done, _ = env_wrapper.step(actions)

        total_rewards[0] += rewards[0]
        total_rewards[1] += rewards[1]

        print(f"Step {step}: actions={actions}, rewards={rewards}, total={total_rewards}")

        if render:
            env_wrapper.env.render()

        state = next_state
        step += 1

    print("=" * 40)
    print(f"Episode finished. Total rewards: {total_rewards}")
    print(f"Steps: {step}")

    # 動画保存（学習済みエージェントのアクションヒストリーをそのまま使う）
    if save_video:
        print(f"Saving video to {video_filename}...")
        # env_wrapper.env は DroneDeliveryEnv インスタンス
        env_wrapper.env.save_render_video(agent_actions_list, filename=video_filename, fps=fps)

    return total_rewards, step, agent_actions_list  # アクションヒストリーも返す


# 環境ラッパーの初期化
env_wrapper = EnvWrapper(grid_size=10, num_agents=2, num_packages=3)

# 評価実行（可視化あり＋動画保存あり）
total_rewards, steps, action_history = evaluate_agent(
    agent,
    env_wrapper,
    render=True,
    max_steps=100,
    save_video=True,
    video_filename="trained_agent_episode.mp4",
    fps=5
)

print(f"Generated action history length: {len(action_history)}")

# 1. アクションヒストリーのみを生成（render=False, save_video=False）
_, _, action_history = evaluate_agent(
    agent,
    env_wrapper,
    render=False,
    max_steps=100,
    save_video=False
)

print(f"Action history length: {len(action_history)}")

# 2. 別の環境インスタンスで動画保存
env_for_video = DroneDeliveryEnv()
env_for_video.reset()
env_for_video.save_render_video(action_history, filename="trained_agent_episode.mp4", fps=5)