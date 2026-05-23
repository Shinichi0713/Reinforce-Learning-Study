import torch
import numpy as np

class MAPPORolloutBuffer:
    def __init__(self, buffer_size, num_agents, obs_shape, action_dim):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        
        # メモリ節約のため obs は uint8 (0-255) で保持
        self.obs = torch.zeros((buffer_size, num_agents, *obs_shape), dtype=torch.uint8)
        
        self.actions = torch.zeros((buffer_size, num_agents))
        self.log_probs = torch.zeros((buffer_size, num_agents))
        self.rewards = torch.zeros((buffer_size, num_agents))
        self.values = torch.zeros((buffer_size, num_agents))
        self.masks = torch.ones((buffer_size, num_agents))
        
        self.step = 0

    def insert(self, obs, actions, log_probs, values, rewards, masks):
        for i, agent_id in enumerate(['first_0', 'second_0']):
            # 入力が float の場合を考慮し、格納時に 0-255 の整数に変換
            # すでに 0-255 ならそのまま、0.0-1.0 なら 255倍して入れる運用も可能です
            curr_obs = obs[agent_id]
            if curr_obs.max() <= 1.0:
                curr_obs = curr_obs * 255.0
            
            self.obs[self.step, i] = curr_obs.to(torch.uint8)
            
            self.actions[self.step, i] = actions[agent_id]
            self.log_probs[self.step, i] = log_probs[agent_id]
            self.values[self.step, i] = values[agent_id]
            self.rewards[self.step, i] = rewards[agent_id]
            self.masks[self.step, i] = masks[agent_id]
            
        self.step = (self.step + 1) % self.buffer_size

    def clear(self):
        self.step = 0

    def get_generator(self, num_mini_batches, advantages, returns):
        batch_size = self.buffer_size * self.num_agents
        mini_batch_size = batch_size // num_mini_batches
        
        # フラット化
        flat_obs = self.obs.view(-1, *self.obs.shape[2:])
        flat_actions = self.actions.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_values = self.values.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        
        ids = torch.eye(self.num_agents).repeat(self.buffer_size, 1, 1).view(-1, self.num_agents)

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            
            # --- ここで正規化を行う ---
            # batchとして切り出した後に float化 & 255割りを行うのが最速です
            yield (
                flat_obs[idx].float() / 255.0,  # 0.0 ~ 1.0 に正規化
                ids[idx],
                flat_actions[idx],
                flat_log_probs[idx],
                flat_values[idx],
                flat_advantages[idx],
                flat_returns[idx]
            )