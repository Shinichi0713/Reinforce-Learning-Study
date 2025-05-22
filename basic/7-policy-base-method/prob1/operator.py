# agentと環境（environment）の相互作用を定義
import torch
import environment, agent

# 学習コード
def train():
    env = environment.PoleGym(is_train=True)
    state_dim = env.env.observation_space.shape[0]
    action_dim = env.env.action_space.n
    policy_net = agent.PolicyNetwork(state_dim, action_dim, is_train=True)
    policy_net.train()
    reward_history = []
    for episode in range(1000):
        states, actions, rewards = env.run_episode(policy_net, policy_net.device)
        returns = env.compute_returns()

        # Policy gradientの更新
        policy_net.update(states, actions, returns)
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
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            if terminated or truncated:
                break
    env.close()


if __name__ == "__main__":
    train()
    play()