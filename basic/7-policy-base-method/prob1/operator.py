# agentと環境（environment）の相互作用を定義
import torch
import environment, agent

def train():
    env = environment.PoleGym(is_train=True)
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.n
    policy_net = agent.PolicyNetwork(state_dim, action_dim)
    policy_net.train()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    reward_history = []
    for episode in range(1000):
        states, actions, rewards = env.run_episode(policy_net, policy_net.device)
        returns = env.compute_returns()

        # Policy gradientの計算
        loss = 0
        for logit_state, action, G in zip(states, actions, returns):
            state_tensor = torch.FloatTensor(logit_state).to(policy_net.device)
            probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
            loss += -log_prob * G  # REINFORCEの損失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = sum(rewards)
        reward_history.append(total_reward)

        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward}")

    env.close()
    policy_net.save()
    draw_graph(reward_history)

def draw_graph(reward_history):
    import matplotlib.pyplot as plt
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.show()


def play():
    env = environment.PoleGym(is_train=False)
    policy_net = agent.PolicyNetwork(env.env.observation_space.shape[0], env.env.action_space.n)
    policy_net.load()
    policy_net.eval()
    state = env.reset()[0]
    with torch.no_grad():
        for _ in range(200):
            env.render()
            action = policy_net.get_action(state)
            # state = torch.tensor(state, dtype=torch.float32)
            action = action.detach().cpu().numpy()
            next_state, reward, terminated, truncated, info, q_weight = env.step(action, state)
            state = next_state
            if terminated or truncated:
                break
    env.close()


if __name__ == "__main__":
    train()
    play()