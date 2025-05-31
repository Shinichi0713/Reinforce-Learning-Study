env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = N_DISCRETE_ACTIONS

q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(10000)
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
target_update_freq = 200

num_episodes = 300

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for t in range(200):
        # ε-greedy action
        if random.random() < epsilon:
            action_idx = random.randint(0, action_dim-1)
        else:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_net(s)
                action_idx = q_values.argmax().item()
        action = [ACTION_SPACE[action_idx]]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action_idx, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # 学習
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target = rewards + gamma * next_q * (1 - dones)
            loss = nn.MSELoss()(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ターゲットネットワークの更新
        if t % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

env.close()
