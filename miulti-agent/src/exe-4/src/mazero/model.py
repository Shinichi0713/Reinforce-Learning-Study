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
        self.device = device  # ここで device 属性を追加

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 2)
        self.reward_head = nn.Linear(hidden_dim, 2)
        self.policy_head = nn.Linear(hidden_dim, 2 * action_dim)

    def forward(self, state):
        # state: (batch, state_dim)
        h = self.encoder(state)
        value = self.value_head(h)
        reward = self.reward_head(h)
        logits = self.policy_head(h)
        logits = logits.view(-1, 2, self.action_dim)
        policy = F.softmax(logits, dim=-1)
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
    """
    MuZero/MAZero 風 MCTS 実装
    - 2エージェントの行動を同時に選択する
    - ネットワークで価値・報酬・方策を予測
    - UCT スコアで選択、訪問回数に基づく行動確率を返す
    """
    def __init__(self, model, env_wrapper, num_simulations=50, discount=0.99, c1=1.25, c2=19652):
        self.model = model
        self.env_wrapper = env_wrapper
        self.num_simulations = num_simulations
        self.discount = discount
        self.c1 = c1
        self.c2 = c2
        self.action_dim = env_wrapper.action_space.n

    def run(self, state):
        """
        state: テンソル (state_dim,)
        戻り値: 各エージェントの行動確率 (2, action_dim)
        """
        root = Node(prior=1.0)
        root.state = state

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
            value = self._expand_and_evaluate(node, parent.state, actions_history)
            self._backpropagate(search_path, value, self.discount)

        # 訪問回数に基づく行動確率を計算
        action_probs = self._get_action_probs(root)
        return action_probs

    def _select_child(self, node):
        """
        UCT スコアに基づいて子ノードを選択
        """
        best_score = -float("inf")
        best_action = None
        best_child = None

        total_visits = sum(child.visit_count for child in node.children.values())

        for action, child in node.children.items():
            # child.state が None にならないようにする
            if child.state is None:
                # 必要に応じて parent.state からコピーするなど
                child.state = node.state.clone() if hasattr(node.state, 'clone') else node.state
            # PUCT スコア（MuZero/AlphaZero 風）
            uct_score = self._compute_uct_score(child, node.prior, total_visits)
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child
            
        return best_action, best_child

    def _compute_uct_score(self, child, parent_prior, total_visits):
        """
        PUCT スコアの計算
        """
        pb_c = np.log((total_visits + self.c2 + 1) / self.c2) + self.c1
        pb_c *= np.sqrt(total_visits) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = child.value()
        return value_score + prior_score

    def _expand_and_evaluate(self, node, parent_state, actions_history):
        # parent_state が None でないことを確認
        if parent_state is None:
            # デフォルトの state を使う（root.state など）
            parent_state = self.root.state
        # 1. ネットワークで価値・報酬・方策を予測
        with torch.no_grad():
            state_tensor = parent_state.unsqueeze(0).to(self.model.device)
            value, reward, policy = self.model(state_tensor)

        value = value.squeeze(0).cpu().numpy()        # (2,)
        reward = reward.squeeze(0).cpu().numpy()     # (2,)
        policy = policy.squeeze(0).cpu().numpy()     # (2, action_dim)

        # 2. すべての行動組み合わせに対して子ノードを作成
        for a1 in range(self.action_dim):
            for a2 in range(self.action_dim):
                # 行動 (a1, a2) に対する事前確率（ネットワーク出力）
                prior_prob = policy[0, a1] * policy[1, a2]  # 簡易的な結合
                child = Node(prior=prior_prob)
                child.reward = reward.mean()  # 簡易的に平均報酬を使用
                child.value_sum = value.mean()  # 簡易的に平均価値を使用
                node.children[(a1, a2)] = child

        # 3. 評価値として平均価値を返す
        return value.mean()

    def _backpropagate(self, search_path, value, discount):
        """
        シミュレーション結果をルートまで伝播
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = node.reward + discount * value  # 割引累積報酬で更新

    def _get_action_probs(self, root):
        """
        訪問回数に基づく行動確率を計算
        """
        action_probs = np.zeros((2, self.action_dim))

        # 各エージェントごとに訪問回数を集計
        for (a1, a2), child in root.children.items():
            action_probs[0, a1] += child.visit_count
            action_probs[1, a2] += child.visit_count

        # 正規化
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


