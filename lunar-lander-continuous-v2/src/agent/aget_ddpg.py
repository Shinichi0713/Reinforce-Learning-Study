
import os 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Ouadminノイズ
class OUActionNoise(object): # Ornstein-Uhlenbeck process -> Temporary correlated noise
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# リプレイバッファ
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size # index of the memory

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward 
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size) # if memory is not full, use mem_cntr
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, terminal


# インスタンス時の次元は状態＋行動
class CriticNet(nn.Module):
    def __init__(self, dim_input):
        super(CriticNet, self).__init__()
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, 'nn_critic_ddpg.pth')
        dim_nn = 256 * 2
        self.fc = nn.Sequential(
            nn.Linear(dim_input, dim_nn),
            nn.LayerNorm(dim_nn),
            nn.ReLU(),
            nn.Linear(dim_nn, 1),
            nn.LayerNorm(1),
            nn.Mish(),
        )
        self.load_checkpoint()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat((state, action), dim=1)
        x = self.fc(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.path_nn)
        print('...critic network saved...')

    def load_checkpoint(self):
        if os.path.isfile(self.path_nn):
            self.load_state_dict(T.load(self.path_nn))
            print('...critic network loaded...')
        else:
            print('...no critic network found...')


class ActorNet(nn.Module):
    def __init__(self, dim_input, dim_actions):
        super(ActorNet, self).__init__()
        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, 'nn_actor_ddpg.pth')
        dim_nn = 256 * 2
        self.fc = nn.Sequential(
            nn.Linear(dim_input, dim_nn),
            nn.LayerNorm(dim_nn),
            nn.ReLU(),
            nn.Linear(dim_nn, dim_actions),
            nn.Tanh(),
        )
        self.load_checkpoint()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc(state)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.path_nn)
        print('...actor network saved...')

    def load_checkpoint(self):
        if os.path.isfile(self.path_nn):
            self.load_state_dict(T.load(self.path_nn))
            print('...actor network loaded...')
        else:
            print('...no actor network found...')


class AgentDdpg(object):
    def __init__(self, alpha, beta, tau, env, batch_size):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.batch_size = batch_size
        
        self.env = env
        self.n_actions = self.env.env.action_space.shape[0]
        self.n_states = self.env.env.observation_space.shape[0]

        self.memory = ReplayBuffer(10000, self.n_states, self.n_actions)
        self.actor = ActorNet(dim_input=self.n_states, dim_actions=self.n_actions)
        self.critic = CriticNet(dim_input=self.n_states + self.n_actions)

        # Target networks
        self.target_actor = ActorNet(dim_input=self.n_states, dim_actions=self.n_actions)
        self.target_critic = CriticNet(dim_input=self.n_states + self.n_actions)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.beta)

        # Noise for exploration
        mu = np.zeros(self.n_actions)
        sigma = 0.15 * np.ones(self.n_actions)
        theta = 0.2
        self.noise = OUActionNoise(mu=mu, sigma=sigma, theta=theta)

    # モデルより行動を選択するメソッド
    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        action = self.actor(state).to(self.actor.device)
        mu_prime = action + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # 経験リプレイ
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        # ターゲットはネットワーク固定
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        # 次の行動出力
        target_actions = self.target_actor.forward(new_state)
        # 状態と行動により価値を出力
        # 次の価値
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        # 現在の価値
        critic_value = self.critic.forward(state, action)

        target = []
        # 教師信号の計算(=ターゲット、未来から生成)
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)   
    
        self.critic.train()
        self.critic.optimizer.zero_grad()
        # ToBeの教師信号と、現在価値のMSE損失を計算
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()
        # アクターの更新
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self):
        
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = self.tau*critic_state_dict[name].clone() + (1-self.tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        
        for name in actor_state_dict:
            actor_state_dict[name] = self.tau*actor_state_dict[name].clone() + (1-self.tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

if __name__ == "__main__":
    critic = CriticNet(dim_input=8)
    actor = ActorNet(dim_input=8, dim_actions=4)
