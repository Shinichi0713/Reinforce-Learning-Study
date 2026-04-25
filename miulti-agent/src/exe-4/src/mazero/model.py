import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import random

# ==============================
# Multi-Agent Delivery Environment
# ==============================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class MAZeroNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, device="cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # エンコーダ（状態 → 隠れ状態）
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 価値ヘッド（隠れ状態 → 価値）
        self.value_head = nn.Linear(hidden_dim, 2)

        # 報酬ヘッド（隠れ状態 → 報酬）
        self.reward_head = nn.Linear(hidden_dim, 2)

        # 方策ヘッド（隠れ状態 → 行動確率）
        self.policy_head = nn.Linear(hidden_dim, 2 * action_dim)

        # 遷移ヘッド（隠れ状態 + 行動 → 次の隠れ状態）
        # 行動は one-hot に変換して入力（簡易版）
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim + 2 * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def encode(self, state):
        # state: (batch, state_dim)
        return self.encoder(state)

    def predict_value_reward_policy(self, hidden_state):
        # hidden_state: (batch, hidden_dim)
        value = self.value_head(hidden_state)          # (batch, 2)
        reward = self.reward_head(hidden_state)        # (batch, 2)
        logits = self.policy_head(hidden_state)         # (batch, 2*action_dim)
        logits = logits.view(-1, 2, self.action_dim)   # (batch, 2, action_dim)
        policy = F.softmax(logits, dim=-1)             # (batch, 2, action_dim)
        return value, reward, policy

    def transition(self, hidden_state, actions):
        # hidden_state: (batch, hidden_dim)
        # actions: (batch, 2) 各エージェントの行動インデックス
        batch_size = hidden_state.size(0)

        # 行動を one-hot に変換
        actions_onehot = torch.zeros(batch_size, 2 * self.action_dim).to(self.device)
        for i in range(2):
            actions_onehot.scatter_(1, actions[:, i].unsqueeze(1) + i * self.action_dim, 1.0)

        # 隠れ状態と行動を結合
        x = torch.cat([hidden_state, actions_onehot], dim=-1)  # (batch, hidden_dim + 2*action_dim)
        next_hidden = self.transition_head(x)                   # (batch, hidden_dim)
        return next_hidden

    def forward(self, state):
        # 互換性のため（既存コード用）
        h = self.encode(state)
        value, reward, policy = self.predict_value_reward_policy(h)
        return value, reward, policy

class Node:
    """MuZero/MAZero 風 MCTS ノード（全エージェント共有）"""
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.prior = prior          # ネットワークが与える事前確率
        self.value_sum = 0.0
        self.children = {}          # key: (a1, a2), value: Node
        self.state = None           # このノードの状態（テンソル）
        self.reward = 0.0           # このノードへの遷移で得た報酬（ネットワーク予測）
        self.hidden_state = None    # モデルが予測した隠れ状態（必要に応じて）

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MAZeroMCTS:
    def __init__(self, model, env_wrapper, num_simulations=50, discount=0.99, c1=1.25, c2=19652, lambda_val=0.8):
        self.model = model
        self.env_wrapper = env_wrapper
        self.num_simulations = num_simulations
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        self.lambda_val = lambda_val  # λ-return 用
        self.action_dim = env_wrapper.action_space.n

    def run(self, state):
        root = Node(prior=1.0)
        root.state = state
        root.hidden_state = self.model.encode(state.unsqueeze(0)).squeeze(0)  # 隠れ状態を保存

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            actions_history = []

            # Selection: UCT で行動を選択しながら木を下る
            while node.expanded():
                action, node = self._select_child(node)
                search_path.append(node)
                actions_history.append(action)

            # Expansion & Evaluation: 葉ノードを展開し、ネットワークで評価
            parent = search_path[-2] if len(search_path) >= 2 else root
            value = self._expand_and_evaluate(node, parent.hidden_state, actions_history)
            self._backpropagate(search_path, value, self.discount)

        # 訪問回数に基づく行動確率を計算
        action_probs = self._get_action_probs(root)
        return action_probs

    def _select_child(self, node):
        best_score = -float("inf")
        best_action = None
        best_child = None

        total_visits = sum(child.visit_count for child in node.children.values())

        for action, child in node.children.items():
            uct_score = self._compute_uct_score(child, node.prior, total_visits)
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child
            
        return best_action, best_child

    def _compute_uct_score(self, child, parent_prior, total_visits):
        pb_c = np.log((total_visits + self.c2 + 1) / self.c2) + self.c1
        pb_c *= np.sqrt(total_visits) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = child.value()
        return value_score + prior_score

    def _expand_and_evaluate(self, node, parent_hidden_state, actions_history):
        # parent_hidden_state: (hidden_dim,)
        # 1. ネットワークで価値・報酬・方策を予測
        with torch.no_grad():
            hidden_state = parent_hidden_state.unsqueeze(0)  # (1, hidden_dim)
            value, reward, policy = self.model.predict_value_reward_policy(hidden_state)

        value = value.squeeze(0).cpu().numpy()        # (2,)
        reward = reward.squeeze(0).cpu().numpy()     # (2,)
        policy = policy.squeeze(0).cpu().numpy()     # (2, action_dim)

        # 2. すべての行動組み合わせに対して子ノードを作成（簡易版：サンプリング推奨）
        for a1 in range(self.action_dim):
            for a2 in range(self.action_dim):
                # 行動 (a1, a2) に対する事前確率
                prior_prob = policy[0, a1] * policy[1, a2]
                child = Node(prior=prior_prob)
                child.reward = reward.mean()  # 簡易的に平均報酬
                child.value_sum = value.mean()

                # 遷移モデルで次の隠れ状態を予測
                actions_tensor = torch.LongTensor([[a1, a2]]).to(self.model.device)
                next_hidden = self.model.transition(hidden_state, actions_tensor).squeeze(0)
                child.hidden_state = next_hidden

                node.children[(a1, a2)] = child

        return value.mean()

    def _backpropagate(self, search_path, value, discount):
        # λ-return 風の価値伝播（簡易版）
        # 実際には n-step return を計算するのが望ましい
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + discount * value

    def _get_action_probs(self, root):
        action_probs = np.zeros((2, self.action_dim))
        for (a1, a2), child in root.children.items():
            action_probs[0, a1] += child.visit_count
            action_probs[1, a2] += child.visit_count
        for i in range(2):
            total = action_probs[i].sum()
            if total > 0:
                action_probs[i] /= total
            else:
                action_probs[i] = np.ones(self.action_dim) / self.action_dim
        return action_probs


class EnvWrapper:
    def __init__(self, grid_size=10, num_agents=2, num_packages=3):
        self.env = DroneDeliveryEnv(
            grid_size=grid_size,
            num_agents=num_agents,
            num_packages=num_packages
        )
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_packages = num_packages
        self.action_space = type('', (), {'n': 7})()  # 7 actions

    def reset(self):
        obs_list = self.env.reset()
        return self._obs_to_tensor(obs_list)

    def step(self, actions):
        obs_list, rewards, done, info = self.env.step(actions)
        return self._obs_to_tensor(obs_list), rewards, done, info

    def _obs_to_tensor(self, obs_list):
        # 観測を1次元ベクトルに変換（簡易版）
        # ここでは「全エージェントの位置」と「全荷物の状態」を連結
        state_vec = []
        for agent_obs in obs_list:
            # エージェント位置
            x, y = agent_obs["agent_pos"]
            state_vec.extend([x, y])
        # 荷物状態（どのエージェントから見ても同じなので1回だけ）
        for p in obs_list[0]["packages"]:
            pick, drop, picked, delivered = p
            state_vec.extend([*pick, *drop, float(picked), float(delivered)])
        return torch.FloatTensor(state_vec)

    @property
    def state_dim(self):
        # 実際に _obs_to_tensor が返す次元数と一致させる
        # 2 agents * 2 (x,y) + num_packages * (2 pick + 2 drop + 2 flags)
        return 2 * 2 + self.num_packages * 6


