# sacによる成功コード
import numpy as np
import torch
import random
from collections import deque, namedtuple
import os
from itertools import chain
import gym

# --- ReplayBuffer ---
import pickle
from archs.trsf_models import Actor, Critic


#https://www.researchgate.net/publication/320296763_Recurrent_Network-based_Deterministic_Policy_Gradient_for_Solving_Bipedal_Walking_Challenge_on_Rugged_Terrains
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from torch.distributions import Normal

EPS = 0.003


class ReplayBuffer:
    """Simle experience replay buffer for deep reinforcement algorithms."""
    def __init__(self, action_size, buffer_size, batch_size, device, seq_len=18):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        state_seq, action, reward, next_state, done = map(np.stack, zip(*batch))
        # state_seq: [batch, 18, 24]
        return (
            torch.from_numpy(state_seq).float(),      # [batch, 18, 24]
            torch.from_numpy(action).float(),
            torch.from_numpy(reward).unsqueeze(1).float(),
            torch.from_numpy(next_state).float(),
            torch.from_numpy(done).unsqueeze(1).float()
        )

    def __len__(self):
        return len(self.memory)

    # def save(self, filepath):
    #     with open(filepath, 'wb') as f:
    #         pickle.dump({
    #             'buffer': list(self.buffer),
    #             'maxlen': self.buffer.maxlen,
    #             'batch_size': self.batch_size
    #         }, f)

    # def load(self, filepath):
    #     with open(filepath, 'rb') as f:
    #         data = pickle.load(f)
    #         self.buffer = deque(data['buffer'], maxlen=data['maxlen'])
    #         self.batch_size = data['batch_size']

# --- SACAgent（ご提示のまま、ReplayBufferの使い方だけ修正） ---
class SACAgent():
    rl_type = 'sac'
    def __init__(self, Actor, Critic, clip_low, clip_high, state_size=24, action_size=4, update_freq=int(1),
            lr=4e-4, weight_decay=0, gamma=0.98, alpha=0.01, tau=0.01, batch_size=64, buffer_size=int(500000), device=None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.update_freq = update_freq

        self.learn_call = int(0)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        else:
            self.device = torch.device(device)

        self.clip_low = torch.tensor(clip_low)
        self.clip_high = torch.tensor(clip_high)

        self.train_actor = Actor(stochastic=True).cpu()
        self.actor_optim = torch.optim.AdamW(self.train_actor.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Actor Net: {sum(p.numel() for p in self.train_actor.parameters())}')
        
        self.train_critic_1 = Critic().cpu()
        self.target_critic_1 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_1, self.target_critic_1) # hard update at the beginning
        self.critic_1_optim = torch.optim.AdamW(self.train_critic_1.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

        self.train_critic_2 = Critic().cpu()
        self.target_critic_2 = Critic().to(self.device).eval()
        self.hard_update(self.train_critic_2, self.target_critic_2) # hard update at the beginning
        self.critic_2_optim = torch.optim.AdamW(self.train_critic_2.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        print(f'Number of paramters of Single Critic Net: {sum(p.numel() for p in self.train_critic_2.parameters())}')
        
        # load
        self.load_ckpt()
        self.memory= ReplayBuffer(action_size= action_size, buffer_size= buffer_size, \
            batch_size= self.batch_size, device=self.device)

        self.mse_loss = torch.nn.MSELoss()
        
    def learn_with_batches(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.learn_one_step()

    def learn_one_step(self):
        if(len(self.memory)>self.batch_size):
            exp=self.memory.sample()
            self.learn(exp)        
            
    def learn(self, exp):
        self.learn_call+=1
        states, actions, rewards, next_states, done = exp
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        done = done.to(self.device)
        #update critic
        with torch.no_grad():
            next_actions, next_entropies = self.train_actor(next_states)
            Q_targets_next_1 = self.target_critic_1(next_states, next_actions)
            Q_targets_next_2 = self.target_critic_2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2) + self.alpha * next_entropies
            Q_targets = rewards + (self.gamma * Q_targets_next * (1-done))
            #Q_targets = rewards + (self.gamma * Q_targets_next)

        Q_expected_1 = self.train_critic_1(states, actions)
        critic_1_loss = self.mse_loss(Q_expected_1, Q_targets)
        #critic_1_loss = torch.nn.SmoothL1Loss()(Q_expected_1, Q_targets)
        
        self.critic_1_optim.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_critic_1.parameters(), 1)
        self.critic_1_optim.step()

        Q_expected_2 = self.train_critic_2(states, actions)   
        critic_2_loss = self.mse_loss(Q_expected_2, Q_targets)
        #critic_2_loss = torch.nn.SmoothL1Loss()(Q_expected_2, Q_targets)
        
        self.critic_2_optim.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_critic_2.parameters(), 1)
        self.critic_2_optim.step()

        #update actor
        actions_pred, entropies_pred = self.train_actor(states)
        Q_pi = torch.min(self.train_critic_1(states, actions_pred), self.train_critic_2(states, actions_pred))
        actor_loss = -(Q_pi + self.alpha * entropies_pred).mean()
        
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.train_actor.parameters(), 1)
        self.actor_optim.step()

        if self.learn_call % self.update_freq == 0:
            self.learn_call = 0        
            #using soft upates
            self.soft_update(self.train_critic_1, self.target_critic_1)
            self.soft_update(self.train_critic_2, self.target_critic_2)

    @torch.no_grad()        
    def get_action(self, state, explore=True):
        #self.train_actor.eval()
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        #with torch.no_grad():
        action, entropy = self.train_actor(state, explore=explore)
        action = action.cpu().data.numpy()[0]
        #self.train_actor.train()
        return action
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save_ckpt(self, model_type, env_type, prefix='last'):
        dir_current = os.path.abspath(os.path.dirname(__file__))
        actor_file = os.path.join(dir_current, "actor.pth")
        critic_1_file = os.path.join(dir_current, "critic_1.pth")
        critic_2_file = os.path.join(dir_current, "critic_2.pth")
        self.train_actor.cpu()
        self.train_critic_1.cpu()
        self.train_critic_2.cpu()
        torch.save(self.train_actor.state_dict(), actor_file)
        torch.save(self.train_critic_1.state_dict(), critic_1_file)
        torch.save(self.train_critic_2.state_dict(), critic_2_file)
        self.train_actor.to(self.device)
        self.train_critic_1.to(self.device)
        self.train_critic_2.to(self.device)

    def load_ckpt(self):
        dir_current = os.path.dirname(os.path.abspath(__file__))
        actor_file = os.path.join(dir_current, "actor.pth")
        critic_1_file = os.path.join(dir_current, "critic_1.pth")
        critic_2_file = os.path.join(dir_current, "critic_2.pth")
        try:
            self.train_actor.load_state_dict(torch.load(actor_file))
            self.train_actor.to(self.device)
        except:
            print("Actor checkpoint cannot be loaded.")
        try:
            self.train_critic_1.load_state_dict(torch.load(critic_1_file))
            self.train_critic_1.to(self.device)
            self.train_critic_2.load_state_dict(torch.load(critic_2_file))
            self.train_critic_2.to(self.device)
        except:
            print("Critic checkpoints cannot be loaded.")              

    def train_mode(self):
        self.train_actor.train()
        self.train_critic_1.train()
        self.train_critic_2.train()

    def eval_mode(self):
        self.train_actor.eval()
        self.train_critic_1.eval()
        self.train_critic_2.eval()

    def freeze_networks(self):
        for p in chain(self.train_actor.parameters(), self.train_critic_1.parameters(), self.train_critic_2.parameters()):
            p.requires_grad = False

    def step_end(self):
        pass

    def episode_end(self):
        pass 

# --- テスト関数のダミー ---
def test(env, agent, render=True, max_t_step=1000, explore=False, n_times=1):
    sum_scores = 0
    for i in range(n_times):
        state = env.reset()
        score = 0
        done=False
        t = int(0)
        while not done and t < max_t_step:
            t += int(1)
            action = agent.get_action(state, explore=explore)
            action = action.clip(min=env.action_space.low, max=env.action_space.high)
            #print(action)
            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward
            if render:
                env.render()
        sum_scores += score
    mean_score = sum_scores/n_times
    print('\rTest Episodes\tMean Score: {:.2f}'.format(mean_score))
    return mean_score

# --- train関数（ご提示のまま） ---
def train(env, agent, n_episodes=8000, model_type='unk', env_type='unk', score_limit=300.0, explore_episode=50, test_f=200, max_t_step=750):
    scores_deque = deque(maxlen=100)
    scores = []
    test_scores = []
    seq_len = 18
    max_score = -np.inf
    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        state_history = deque(maxlen=seq_len)
        state_history.append(state)
        done = False
        score = 0.0
        while not done:
            # 履歴をパディングしてActorに渡す
            action = get_action_with_padding(agent, state_history, state_dim, seq_len, agent.device)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 次の履歴を用意
            next_state_history = deque(state_history, maxlen=seq_len)
            next_state_history.append(next_state)

            if len(state_history) == seq_len:
                agent.memory.add(
                    list(state_history),         # 現在の履歴
                    action,
                    reward,
                    list(next_state_history),    # 次の履歴
                    done
                )

            state_history.append(next_state)
            state = next_state
            state = next_state
            score += reward
            agent.step_end()
            #if i_episode>explore_episode:
            #    env.render()

        if i_episode>explore_episode:
            agent.episode_end()
            agent.learn_one_step()

        scores_deque.append(score)
        avg_score_100 = np.mean(scores_deque)
        scores.append((i_episode, score, avg_score_100))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, avg_score_100, score), end="")

        if i_episode % 100 == 0:
        #     print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        #     agent.eval_mode() # test in eval mode.
        #     test_score = test(env, agent, render=False, n_times=20)
        #     test_scores.append((i_episode, test_score))
            agent.save_ckpt(model_type, env_type,'ep'+str(int(i_episode)))
        #     if avg_score_100>score_limit:
        #         break
            agent.train_mode() # when the test done, come back to train mode.

    return np.array(scores).transpose()
    # return np.array(scores).transpose(), np.array(test_scores).transpose()


def get_action_with_padding(agent, state_history, state_dim, seq_len, device, explore=True):
    # state_history: deque([state, ...], maxlen=seq_len)
    # state_dim: 24
    # seq_len: 18
    history = list(state_history)
    # パディング
    if len(history) < seq_len:
        pad = [history[0]] * (seq_len - len(history))  # 最初の状態で埋める
        history = pad + history
    states_arr = np.stack(history)  # [seq_len, state_dim]
    states_arr = states_arr[np.newaxis, ...]  # [1, seq_len, state_dim]
    states_tensor = torch.from_numpy(states_arr).float().to(device)
    action, _ = agent.train_actor(states_tensor, explore=explore)
    return action.cpu().data.numpy()[0]

# --- 実行例 ---
if __name__ == "__main__":
    # 環境・エージェントの準備
    env = gym.make('BipedalWalkerHardcore-v3')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = SACAgent(
        Actor,
        Critic,
        clip_low=-1, clip_high=1, state_size=state_dim, action_size=action_dim
    )

    # 1エピソードだけ動作確認
    train(env, agent, n_episodes=10000, model_type='dummy', env_type='dummy', score_limit=2.0, explore_episode=0, test_f=1, max_t_step=10)
