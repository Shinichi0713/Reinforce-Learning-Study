# agentと環境（environment）の相互作用を定義
import torch
import environment, agent

def train():
    env = environment.PoleGym()
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.n
    policy_net = agent.PolicyNetwork(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-2)
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

        # for logit_state, action, G in zip(states, actions, returns):
        #     state_tensor = torch.FloatTensor(logit_state).to(device)
        #     probs = policy_net(state_tensor)
        #     dist = torch.distributions.Categorical(probs)
        #     log_prob = dist.log_prob(action)
        #     loss += -log_prob * G  # REINFORCEの損失

if __name__ == "__main__":
    train()