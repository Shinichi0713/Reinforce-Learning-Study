
import numpy as np
import time, os, random
import torch
import torch.nn as nn
from statistics import mean, median
import matplotlib.pyplot as plt
from collections import deque
from environment import Env
from agent import Actor, Critic, ReplayBuffer, AgentImitation


def collect_experience():
    env = Env(is_train=True)
    state_dim, action_dim = env.get_dims()

    agent_expert = Actor(state_dim, action_dim)
    agent_expert.eval()  # エキスパートエージェントは評価モード

    num_episodes = 500
    reward_history = []
    experience_state = []
    experience_action = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        # 倒れるまでアクション
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                probs_action = agent_expert(state_tensor).cpu().squeeze(0).numpy()

            action = np.random.choice(action_dim, p=probs_action)
            next_state, reward, done = env.step(action)
            episode_reward += reward
            # 経験の蓄積
            experience_state.append(state)
            experience_action.append(action)
            state = next_state

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}")
        reward_history.append(episode_reward)

    # 経験の保存
    dir_store = os.path.join(os.path.dirname(__file__), 'experience')
    if not os.path.exists(dir_store):
        os.makedirs(dir_store)
    np.save(os.path.join(dir_store, 'experience_state.npy'), np.array(experience_state, dtype=np.float32))
    np.save(os.path.join(dir_store, 'experience_action.npy'), np.array(experience_action, dtype=np.int64))


# DDQN学習本体
def train_ddqn(env, episodes=300, batch_size=64):
    state_dim, action_dim = env.get_dims()

    policy_net = AgentImitation(state_dim, action_dim)
    target_net = AgentImitation(state_dim, action_dim)
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=2e-4)
    buffer = ReplayBuffer(10000)
    gamma = 0.99
    epsilon_start = 0.90
    epsilon_final = 0.01
    epsilon_decay = episodes / 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    target_net.to(device)

    def select_action(state, epsilon):
        if random.random() < epsilon:
            return random.randrange(action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state)
            return q_values.argmax().item()

    epsilon = epsilon_start
    reward_history = []
    loss_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        loss_total = 0.0
        count_total = 0
        done = False

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                # サンプリング
                states, actions, rewards_, next_states, dones = buffer.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_ = torch.FloatTensor(rewards_).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Q値計算
                q_values = policy_net(states).gather(1, actions)
                # Double DQN: 次の行動はpolicy_net、Q値はtarget_net
                next_actions = policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = target_net(next_states).gather(1, next_actions)
                expected_q = rewards_ + gamma * next_q_values

                loss = nn.MSELoss()(q_values, expected_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
                count_total += 1

        # ターゲットネットの更新
        soft_update(target_net, policy_net, tau=0.01)

        # ε-greedyの減衰
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * episode / epsilon_decay)
        if count_total > 0:
            loss_total /= count_total
            loss_history.append(loss_total)
            reward_history.append(total_reward)

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
            policy_net.save_nn()

    env.close()
    policy_net.save_nn()
    return policy_net, reward_history, loss_history

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

def train_imitation():
    env = Env(is_train=True)
    state_dim, action_dim = env.get_dims()
    agent_imitation = AgentImitation(state_dim, action_dim)
    agent_optim = torch.optim.Adam(agent_imitation.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_episodes = 5000
    num_episodes_imitation = 50
    reward_history = []
    loss_history_imitate = []

    # 経験のロード
    dir_store = os.path.join(os.path.dirname(__file__), 'experience')
    path_state = os.path.join(dir_store, 'experience_state.npy')
    path_action = os.path.join(dir_store, 'experience_action.npy')
    if os.path.exists(path_state) and os.path.exists(path_action):
        experience_state = np.load(path_state)
        experience_action = np.load(path_action)
    else:
        assert False, "Experience data not found. Please run collect_experience() first."

    # 模倣学習のための教師あり学習
    for epoch in range(num_episodes_imitation):
        idx = np.random.permutation(len(experience_state))
        states_shuffled = torch.tensor(experience_state[idx]).to(agent_imitation.device)
        actions_shuffled = torch.tensor(experience_action[idx]).to(agent_imitation.device)

        logits = agent_imitation(states_shuffled)
        loss = criterion(logits, actions_shuffled)

        agent_optim.zero_grad()
        loss.backward()
        agent_optim.step()

        loss_history_imitate.append(loss.item())
        reward_history.append(0.0)  # 模倣学習では報酬は計算しない
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
    agent_imitation.save_nn()

    policy_net, reward_history_ddqn, loss_history_ddqn = train_ddqn(env, episodes=num_episodes-num_episodes_imitation, batch_size=64)

    loss_history = loss_history_imitate + loss_history_ddqn
    reward_history += reward_history_ddqn
    dir_current = os.path.dirname(os.path.abspath(__file__))
    write_log(f"{dir_current}/reward_history_imitation.txt", reward_history)
    write_log(f"{dir_current}/loss_history_imitation.txt", loss_history)

def write_log(path, data):
    with open(path, 'w') as f:
        for d in data:
            f.write(str(d) + '\n')

# 評価
def evaluate():
    env = Env(is_train=False)
    state_dim, action_dim = env.get_dims()
    actor = AgentImitation(state_dim, action_dim)
    actor.eval()
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            action_prob = actor(state_tensor)[0]
        action = torch.max(action_prob, dim=-1)[1].item()  # 最大の確率のアクションを選択
        state, reward, done = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.002)

    print(f"Total reward in evaluation: {total_reward}")
    env.close()


if __name__ == "__main__":
    # collect_experience()
    train_imitation()
    evaluate()