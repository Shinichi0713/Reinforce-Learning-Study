import torch

class MAPPOMemory:
    def __init__(self):
        self.obs = []        # 各エージェントの個別観測 (T, NumAgents, ObsDim)
        self.states = []     # 集中Critic用の全体状態 (T, ObsDim * NumAgents)
        self.actions = []    # 各エージェントの行動
        self.log_probs = []  # 各エージェントの行動ログ確率
        self.rewards = []    # 各エージェントの報酬
        self.dones = []      # 終了フラグ
        self.h_actors = []   # 各エージェントのGRU隠れ状態 (初期状態のみ保存)
        self.h_critics = []  # CriticのGRU隠れ状態 (初期状態のみ保存)

    def store(self, obs, state, action, log_prob, reward, done):
        self.obs.append(obs)
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.obs, self.states, self.actions, self.log_probs = [], [], [], []
        self.rewards, self.dones, self.h_actors, self.h_critics = [], [], [], []