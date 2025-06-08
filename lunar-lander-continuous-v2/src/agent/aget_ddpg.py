
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



class CriticNet(nn.Module):
    def __init__(self, dim_input, dim_actions):
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

if __name__ == "__main__":
    critic = CriticNet(dim_input=8, dim_actions=4)
    actor = ActorNet(dim_input=8, dim_actions=4)
    