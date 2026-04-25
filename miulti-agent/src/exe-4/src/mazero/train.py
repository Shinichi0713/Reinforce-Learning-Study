class MAZeroAgent:
    def __init__(self, model, mcts, optimizer, device="cpu"):
        self.model = model.to(device)
        self.mcts = mcts
        self.optimizer = optimizer
        self.device = device

    def select_action(self, state, temperature=1.0):
        action_probs = self.mcts.run(state)
        actions = []
        for i in range(2):
            probs = action_probs[i]
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs /= probs.sum()
            action = np.random.choice(self.mcts.action_dim, p=probs)
            actions.append(action)
        return actions

    def update(self, states, target_values, target_rewards, target_policies, advantages=None):
        """
        AWPO 風のアドバンテージ重み付けを追加
        advantages: (batch, 2) 各エージェントのアドバンテージ
        """
        self.model.train()
        self.optimizer.zero_grad()

        states = torch.stack(states).to(self.device)
        target_values = torch.FloatTensor(np.array(target_values)).to(self.device)
        target_rewards = torch.FloatTensor(np.array(target_rewards)).to(self.device)
        target_policies = torch.FloatTensor(np.array(target_policies)).to(self.device)

        pred_values, pred_rewards, pred_policies = self.model(states)

        # 価値損失（MSE）
        value_loss = F.mse_loss(pred_values, target_values)

        # 報酬損失（MSE）
        reward_loss = F.mse_loss(pred_rewards, target_rewards)

        # ポリシー損失（AWPO 風：アドバンテージ重み付け）
        if advantages is None:
            # アドバンテージがなければ通常のクロスエントロピー
            policy_loss = -torch.sum(target_policies * torch.log(pred_policies + 1e-8)) / len(states)
        else:
            # advantages: (batch, 2)
            advantages = torch.FloatTensor(advantages).to(self.device)
            # 各エージェントごとに重み付け
            policy_loss = 0.0
            for i in range(2):
                # アドバンテージが正の行動を強調
                weight = torch.clamp(advantages[:, i], min=0.0) + 1e-8
                # クロスエントロピーに重み付け
                ce = -torch.sum(target_policies[:, i, :] * torch.log(pred_policies[:, i, :] + 1e-8), dim=-1)
                policy_loss += torch.mean(weight * ce)
            policy_loss /= 2  # エージェント数で平均

        loss = value_loss + reward_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

def train_mazero(num_episodes=1000, num_simulations=50, lambda_val=0.8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_wrapper = EnvWrapper(grid_size=10, num_agents=2, num_packages=3)
    state_dim = env_wrapper.state_dim
    action_dim = env_wrapper.action_space.n

    model = MAZeroNet(state_dim, action_dim, hidden_dim=128, device=device)
    mcts = MAZeroMCTS(model, env_wrapper, num_simulations=num_simulations, lambda_val=lambda_val)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    agent = MAZeroAgent(model, mcts, optimizer, device=device)

    replay_buffer = deque(maxlen=10000)

    for episode in range(num_episodes):
        state = env_wrapper.reset()
        episode_states = []
        episode_rewards = []
        episode_mcts_probs = []
        episode_values = []  # ネットワークの価値予測を記録

        done = False
        while not done:
            # MCTS の行動確率を取得
            mcts_probs = mcts.run(state)
            actions = agent.select_action(state, temperature=1.0)

            next_state, rewards, done, _ = env_wrapper.step(actions)

            # ネットワークの価値予測を記録（アドバンテージ計算用）
            with torch.no_grad():
                state_tensor = state.unsqueeze(0).to(device)
                value, _, _ = model(state_tensor)
                value = value.squeeze(0).cpu().numpy()  # (2,)

            episode_states.append(state)
            episode_rewards.append(rewards)
            episode_mcts_probs.append(mcts_probs)
            episode_values.append(value)

            state = next_state

        # λ-return 風のターゲット価値計算（簡易版）
        # 実際には n-step return を計算するのが望ましい
        returns = np.zeros((len(episode_rewards), 2))
        R = np.zeros(2)
        for t in reversed(range(len(episode_rewards))):
            R = episode_rewards[t] + mcts.discount * R
            returns[t] = R

        # アドバンテージ計算（実際の return - 予測価値）
        advantages = np.zeros((len(episode_rewards), 2))
        for t in range(len(episode_rewards)):
            advantages[t] = returns[t] - episode_values[t]

        # 経験をバッファに保存
        for t in range(len(episode_states)):
            replay_buffer.append((
                episode_states[t],
                returns[t],           # target_value
                episode_rewards[t],   # target_reward
                episode_mcts_probs[t],# target_policy
                advantages[t]        # advantage
            ))

        # バッチ学習
        if len(replay_buffer) >= 32:
            batch = random.sample(replay_buffer, 32)
            states, t_values, t_rewards, t_policies, t_advantages = zip(*batch)
            loss = agent.update(states, t_values, t_rewards, t_policies, t_advantages)
            if episode % 100 == 0:
                print(f"Episode {episode}, Loss: {loss:.4f}")

    return agent