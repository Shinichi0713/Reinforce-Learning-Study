episodes = 5000
win_count = 0

for ep in range(episodes):
    state = env.reset()
    agent.log_probs = []
    agent.rewards = []

    while not env.is_terminal():
        current_player = env.state.current_player()
        legal_actions = list(env.legal_actions())

        if current_player == 0:
            action = agent.act(env.state, legal_actions)
        else:
            action = random.choice(legal_actions)

        state = env.step(action)

    # 報酬の設定（プレイヤー0視点）
    returns = state.returns()
    reward = returns[0]
    # 報酬は float のまま保持（勾配計算に必要なのは log_prob のみ）
    agent.rewards = [reward] * len(agent.log_probs)

    agent.update()

    if reward == 1:
        win_count += 1

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}, Win rate: {win_count / (ep+1):.3f}")
