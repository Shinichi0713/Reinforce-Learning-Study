import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

# --- 1. Model 定義 (動的シェイプ対応) ---
class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AlphaZeroNet, self).__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * h * w, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(h * w, 64),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        p = self.policy_head(x)
        v = self.value_head(x)
        return F.softmax(p, dim=1), v

# --- 2. Evaluator 定義 ---
class AlphaZeroEvaluator(mcts.Evaluator):
    def __init__(self, model, input_shape, device="cpu"):
        self.model = model
        self.input_shape = input_shape
        self.device = device

    def evaluate(self, state):
        obs = torch.FloatTensor(state.observation_tensor()).reshape(1, *self.input_shape).to(self.device)
        with torch.no_grad():
            probs, value = self.model(obs)
        probs = probs.cpu().numpy()[0]
        return value.item(), {a: probs[a] for a in state.legal_actions()}

    def prior(self, state):
        return self.evaluate(state)

# --- 3. 学習メインループ ---
def train_alphazero():
    board_size = 9
    game = pyspiel.load_game(f"go(board_size={board_size})")
    num_actions = board_size * board_size + 1
    input_shape = game.observation_tensor_shape()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet(input_shape, num_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    memory = []

    for episode in range(5):
        state = game.new_initial_state()
        evaluator = AlphaZeroEvaluator(model, input_shape, device)

        episode_data = []
        print(f"Episode {episode} starting...")

        while not state.is_terminal():
            # MCTSBot の代わりに SearchNode を直接使用して探索
            # uct_c=1.5, simulations=100
            root = mcts.SearchNode(None, 0, 0, 0)
            for _ in range(100):
                # 内部的な MCTS 探索の 1 ステップを実行
                mcts.mcts_step(state, root, 1.5, evaluator, np.random.RandomState())

            # 訪問回数から方策ターゲットを作成
            policy_target = np.zeros(num_actions)
            for action, child in root.children.items():
                policy_target[action] = child.explore_count

            total_visits = np.sum(policy_target)
            if total_visits > 0:
                policy_target /= total_visits
            else:
                legal_actions = state.legal_actions()
                policy_target[legal_actions] = 1.0 / len(legal_actions)

            # 現在の情報を保存
            episode_data.append((state.observation_tensor(), policy_target, state.current_player()))

            # 行動選択
            action = np.random.choice(num_actions, p=policy_target)
            state.apply_action(action)

        # ゲーム終了時の報酬を全ステップに適用
        returns = state.returns()
        for obs, p_target, player in episode_data:
            memory.append((obs, p_target, returns[player]))

        # 学習ステップ
        if len(memory) >= 32:
            indices = np.random.choice(len(memory), 32, replace=False)
            batch = [memory[i] for i in indices]

            obs_b = torch.FloatTensor([x[0] for x in batch]).reshape(32, *input_shape).to(device)
            p_b = torch.FloatTensor([x[1] for x in batch]).to(device)
            v_b = torch.FloatTensor([x[2] for x in batch]).reshape(32, 1).to(device)

            optimizer.zero_grad()
            p_pred, v_pred = model(obs_b)

            loss_v = F.mse_loss(v_pred, v_b)
            loss_p = -torch.mean(torch.sum(p_b * torch.log(p_pred + 1e-8), dim=1))
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()

            print(f"  Step Loss: {loss.item():.4f}")

    return model
