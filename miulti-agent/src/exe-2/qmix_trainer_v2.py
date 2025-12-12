import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# --- 設定の更新 (より深いモデルに合わせて調整) ---
BATCH_SIZE = 128        # モデルが大きくなったのでバッチサイズを少し増やす
HIDDEN_DIM_AGENT = 128  # エージェントの隠れ層のサイズ
HIDDEN_DIM_MIXER = 64   # ミキサーの隠れ層のサイズ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# 1. ネットワーク定義 (Deep Version)
# ----------------------------------------------------

class DeepAgentNet(nn.Module):
    """
    より深く、厚いエージェントネットワーク
    構造: Input -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(Output)
    """
    def __init__(self, input_dim, output_dim):
        super(DeepAgentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM_AGENT),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_AGENT, HIDDEN_DIM_AGENT), # 追加された層
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_AGENT, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DeepQMixer(nn.Module):
    """
    ハイパーネットワークを多層化したミキサー
    状態(s)から重み(W)を生成する際、非線形な変換を挟むことで表現力を向上。
    """
    def __init__(self, num_agents, state_dim, hidden_dim):
        super(DeepQMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # ハイパーネットワークの隠れ層サイズ
        hyper_net_hidden = 64

        # --- ハイパーネットワーク W1 (2層構造化) ---
        # State -> Hidden -> ReLU -> Weights (Num_agents * Hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_net_hidden),
            nn.ReLU(),
            nn.Linear(hyper_net_hidden, hidden_dim * num_agents)
        )
        # バイアス B1 (1層のままですがサイズ調整)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # --- ハイパーネットワーク W2 (2層構造化) ---
        # State -> Hidden -> ReLU -> Weights (Hidden_dim * 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_net_hidden),
            nn.ReLU(),
            nn.Linear(hyper_net_hidden, hidden_dim)
        )
        # バイアス B2 (State -> 1層 -> ReLU -> 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_net_hidden),
            nn.ReLU(),
            nn.Linear(hyper_net_hidden, 1)
        )

    def forward(self, agent_qs, states):
        # agent_qs: (batch, num_agents, 1)
        # states: (batch, state_dim)
        batch_size = agent_qs.size(0)

        # --- W1 の計算 ---
        w1 = torch.abs(self.hyper_w1(states)) # 非負制約
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        
        # --- Q * W1 ---
        agent_qs = agent_qs.view(batch_size, self.num_agents, 1)
        hidden = torch.bmm(agent_qs.transpose(1, 2), w1) # (B, 1, H)

        # --- B1 ---
        b1 = self.hyper_b1(states).view(batch_size, 1, self.hidden_dim)
        hidden = F.elu(hidden + b1) # ReLUの代わりにELUを使うことも深層では有効

        # --- W2 の計算 ---
        w2 = torch.abs(self.hyper_w2(states)) # 非負制約
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        
        # --- Output ---
        q_tot = torch.bmm(hidden, w2) # (B, 1, 1)
        
        # --- B2 ---
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        
        q_tot = q_tot + b2
        return q_tot.squeeze(-1) # (B, 1)


# ----------------------------------------------------
# 3. QMIXエージェントクラス (Deep版を使用するように更新)
# ----------------------------------------------------

class QMixAgent:
    def __init__(self, env: 'WarehouseEnv', state_dim):
        self.env = env
        self.num_agents = env.num_agents
        self.action_space = env.action_space
        self.state_dim = state_dim
        
        # --- 変更点: DeepAgentNet と DeepQMixer を使用 ---
        self.agent_nets = nn.ModuleList([
            DeepAgentNet(self._get_agent_input_dim(), self.action_space).to(device) 
            for _ in range(self.num_agents)
        ])
        
        self.mixer = DeepQMixer(
            self.num_agents, self.state_dim, hidden_dim=HIDDEN_DIM_MIXER
        ).to(device)

        # Target Networks
        self.target_agent_nets = nn.ModuleList([
            DeepAgentNet(self._get_agent_input_dim(), self.action_space).to(device) 
            for _ in range(self.num_agents)
        ])
        
        self.target_mixer = DeepQMixer(
            self.num_agents, self.state_dim, hidden_dim=HIDDEN_DIM_MIXER
        ).to(device)
        
        self.update_target_networks()
        
        # Optimizers (学習率などは変更なし)
        # 必要であれば LR_AGENT = 0.0001 程度に下げても良い（モデルが深い場合）
        self.optimizer_agents = optim.Adam(self._get_agent_params(), lr=0.0005)
        self.optimizer_mixer = optim.Adam(self.mixer.parameters(), lr=0.001)
        
        # リプレイメモリは以前の定義(QMixReplayMemory)を使用
        # ここでは定義済みのものを使う前提
        self.memory = QMixReplayMemory(50000) 
        self.steps_done = 0

    # ... (以下のメソッドは変更なし: そのままコピーしてください) ...
    def _get_agent_input_dim(self):
        return 2 + 1 + 3 # NUM_ORDERSは3と仮定

    def _get_agent_params(self):
        params = []
        for net in self.agent_nets:
            params.extend(list(net.parameters()))
        return params

    def update_target_networks(self):
        for policy_net, target_net in zip(self.agent_nets, self.target_agent_nets):
            target_net.load_state_dict(policy_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def preprocess_state(self, obs):
        # 以前のコードと同じ実装を使用してください
        # ここでは省略しますが、必ず実装が必要です
        global_state = []
        # ... (前回のpreprocess_stateの中身) ...
        # (簡単のため再実装例)
        order_mask = [0.0] * 3
        remaining = obs[0][2]
        for oid in remaining: order_mask[oid] = 1.0
        
        for i in range(self.num_agents):
            (px, py), holding, _ = obs[i]
            global_state.extend([px/9.0, py/9.0, float(holding)])
        global_state.extend(order_mask)
        
        global_tensor = torch.tensor([global_state], dtype=torch.float32, device=device)
        
        local_states = []
        for i in range(self.num_agents):
            (px, py), holding, _ = obs[i]
            lin = [px/9.0, py/9.0, float(holding)]
            lin.extend(order_mask)
            local_states.append(torch.tensor([lin], dtype=torch.float32, device=device))
            
        return global_tensor, local_states

    def select_action(self, global_state, local_states):
        # 以前のコードと同じ
        eps = 0.05 + (1.0 - 0.05) * np.exp(-1. * self.steps_done / 5000)
        self.steps_done += 1
        actions = {}
        if random.random() > eps:
            with torch.no_grad():
                for i in range(self.num_agents):
                    q = self.agent_nets[i](local_states[i])
                    actions[i] = q.max(1)[1].item()
        else:
            actions = {i: random.randrange(self.action_space) for i in range(self.num_agents)}
        return actions

    def optimize_model(self):
        # 基本ロジックは以前と同じですが、変数がDeepモデル用に適応されます
        # コードが長くなるため、前回の optimize_model をそのまま使用してください。
        # ネットワークの構造が変わっても入出力のインターフェースは維持しているため動作します。
        
        if len(self.memory) < BATCH_SIZE: return 0.0
        transitions = self.memory.sample(BATCH_SIZE)
        
        # ... (前回の optimize_model の中身をここに配置) ...
        # バッチ処理 -> Q計算 -> Loss計算 -> Backward
        
        # 簡略化のため中身は省略しますが、必ず前回のコードを使用してください
        return 0.0 # ダミーリターン

    def save_model(self, path):
        # 以前と同じ
        state = {f'agent_{i}': n.state_dict() for i, n in enumerate(self.agent_nets)}
        state['mixer'] = self.mixer.state_dict()
        torch.save(state, path)