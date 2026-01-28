import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MAPPOMemory:
    def __init__(self):
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

        self.h_actors = []  # 各エージェントのActor初期隠れ状態
        self.h_critcs = []  # Criticの初期隠れ状態

    def store(self, obs, state, action, log_prob, reward, done):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        """学習後にメモリを空にする"""
        self.obs = []
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.h_actors = []
        self.h_critics = []

    def get_batch(self):
        """保存されたリストをテンソルに変換して返す（デバッグや拡張用）"""
        return {
            'obs': torch.stack(self.obs),
            'states': torch.stack(self.states),
            'actions': torch.stack(self.actions),
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.stack(self.rewards),
            'dones': torch.tensor(self.dones, dtype=torch.float)
        }
