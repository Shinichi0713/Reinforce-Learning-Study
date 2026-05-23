import torch
import numpy as np

class MAPPORolloutBuffer:
    def __init__(self, buffer_size, num_agents, obs_shape, action_dim):
        """
        buffer_size: 1回の学習までに溜めるステップ数
        num_agents: エージェント数 (Wizard of Worなら 2)
        obs_shape: 画像のサイズ (3, 210, 160)
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        
        # データの格納場所 (PyTorchテンソルで確保)
        # すべて [ステップ数, エージェント数, 次元] の形に揃える
        self.obs = torch.zeros((buffer_size, num_agents, *obs_shape))
        self.actions = torch.zeros((buffer_size, num_agents))
        self.log_probs = torch.zeros((buffer_size, num_agents))
        self.rewards = torch.zeros((buffer_size, num_agents))
        self.values = torch.zeros((buffer_size, num_agents))
        self.masks = torch.ones((buffer_size, num_agents)) # 終了判定用 (dones)
        
        self.step = 0

    def insert(self, obs, actions, log_probs, values, rewards, masks):
        """
        1ステップ分の全エージェントデータを一括挿入
        obs: {agent_id: tensor} のような辞書、または [num_agents, C, H, W] のテンソル
        """
        # ここでは辞書からテンソルに変換して格納する例
        for i, agent_id in enumerate(['first_0', 'second_0']):
            self.obs[self.step, i] = obs[agent_id]
            self.actions[self.step, i] = actions[agent_id]
            self.log_probs[self.step, i] = log_probs[agent_id]
            self.values[self.step, i] = values[agent_id]
            self.rewards[self.step, i] = rewards[agent_id]
            self.masks[self.step, i] = masks[agent_id]
            
        self.step = (self.step + 1) % self.buffer_size

    def clear(self):
        """学習後にポインタをリセット"""
        self.step = 0

    def get_generator(self, num_mini_batches, advantages, returns):
        """
        学習用のミニバッチを生成するジェネレータ
        """
        batch_size = self.buffer_size * self.num_agents
        mini_batch_size = batch_size // num_mini_batches
        
        # データを平坦化 (flatten) してシャッフル
        # [Step, Agent, ...] -> [Step * Agent, ...]
        flat_obs = self.obs.view(-1, *self.obs.shape[2:])
        flat_actions = self.actions.view(-1)
        flat_log_probs = self.log_probs.view(-1)
        flat_values = self.values.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        
        # エージェントIDのOne-hotもフラットに作成
        # [Step, Agent, ID_dim]
        ids = torch.eye(self.num_agents).repeat(self.buffer_size, 1, 1).view(-1, self.num_agents)

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            yield (
                flat_obs[idx],
                ids[idx],
                flat_actions[idx],
                flat_log_probs[idx],
                flat_values[idx],
                flat_advantages[idx],
                flat_returns[idx]
            )