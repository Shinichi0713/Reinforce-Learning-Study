class MAZeroAgent:
    def __init__(self, model, mcts, optimizer, device="cpu"):
        self.model = model.to(device)
        self.mcts = mcts
        self.optimizer = optimizer
        self.device = device

    def select_action(self, state, temperature=1.0):
        """
        MCTS の行動確率に基づいて行動をサンプリング
        
        Parameters
        ----------
        state : torch.Tensor
            現在の状態
        temperature : float
            温度パラメータ（1.0 でそのまま、0に近いと greedy に近づく）
        """
        action_probs = self.mcts.run(state)  # shape: (2, action_dim)

        actions = []
        for i in range(2):
            probs = action_probs[i]
            # 温度でスケーリング
            if temperature != 1.0:
                probs = np.power(probs, 1.0 / temperature)
                probs /= probs.sum()
            # サンプリング
            action = np.random.choice(self.mcts.action_dim, p=probs)
            actions.append(action)

        return actions

    # update メソッドはそのまま
    def update(self, states, target_values, target_rewards, target_policies):
        self.model.train()
        self.optimizer.zero_grad()

        # リストをまとめて numpy.ndarray → torch.Tensor に変換
        states = torch.stack(states).to(self.device)
        target_values = torch.FloatTensor(np.array(target_values)).to(self.device)
        target_rewards = torch.FloatTensor(np.array(target_rewards)).to(self.device)
        target_policies = torch.FloatTensor(np.array(target_policies)).to(self.device)

        pred_values, pred_rewards, pred_policies = self.model(states)

        value_loss = F.mse_loss(pred_values, target_values)
        reward_loss = F.mse_loss(pred_rewards, target_rewards)
        policy_loss = -torch.sum(target_policies * torch.log(pred_policies + 1e-8)) / len(states)

        loss = value_loss + reward_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

def train_mazero(num_episodes=1000, num_simulations=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_wrapper = EnvWrapper(grid_size=10, num_agents=2, num_packages=3)
    state_dim = env_wrapper.state_dim
    action_dim = env_wrapper.action_space.n

    # ここで device を渡す
    model = MAZeroNet(state_dim, action_dim, hidden_dim=128, device=device)
    mcts = MAZeroMCTS(model, env_wrapper, num_simulations=num_simulations)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    agent = MAZeroAgent(model, mcts, optimizer, device=device)

    replay_buffer = deque(maxlen=10000)

    for episode in range(num_episodes):
        state = env_wrapper.reset()
        episode_states = []
        episode_rewards = []
        episode_mcts_probs = []  # MCTS の行動確率を記録

        done = False
        while not done:
            # MCTS の行動確率を取得（サンプリングは別途）
            mcts_probs = mcts.run(state)
            actions = agent.select_action(state, temperature=1.0)  # サンプリング

            next_state, rewards, done, _ = env_wrapper.step(actions)

            episode_states.append(state)
            episode_rewards.append(rewards)
            episode_mcts_probs.append(mcts_probs)  # ターゲット方策として使用

            state = next_state

        # n-step bootstrapping による価値ターゲット（簡易版）
        returns = np.zeros((len(episode_rewards), 2))
        R = np.zeros(2)
        for t in reversed(range(len(episode_rewards))):
            R = episode_rewards[t] + mcts.discount * R
            returns[t] = R

        for t in range(len(episode_states)):
            target_value = returns[t]  # shape: (2,)
            target_reward = episode_rewards[t]  # shape: (2,)
            target_policy = episode_mcts_probs[t]  # shape: (2, action_dim)

            replay_buffer.append((
                episode_states[t],
                target_value,
                target_reward,
                target_policy
            ))

        # バッチ学習
        if len(replay_buffer) >= 32:
            batch = random.sample(replay_buffer, 32)
            states, t_values, t_rewards, t_policies = zip(*batch)
            loss = agent.update(states, t_values, t_rewards, t_policies)
            if episode % 100 == 0:
                print(f"Episode {episode}, Loss: {loss:.4f}")

    return agent