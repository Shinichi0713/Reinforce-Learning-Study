import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QNetwork(nn.Module):
    """各エージェント用のQネットワーク（zを入力に追加）"""
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, z):
        # x: (batch_size, obs_dim), z: (batch_size, z_dim) を想定
        # z が3次元の場合（(batch_size, num_agents, z_dim)）、エージェント次元を削除
        if z.dim() == 3:
            z = z[:, 0, :]  # エージェント0のzを使用（全エージェントで同じzを想定）
        xz = torch.cat([x, z], dim=-1)
        xz = F.relu(self.fc1(xz))
        xz = F.relu(self.fc2(xz))
        q_values = self.fc3(xz)
        return q_values

class MixingNetwork(nn.Module):
    """QMIXのMixing Network（zを入力に追加）"""
    def __init__(self, num_agents, state_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # 重みとバイアスを生成するネットワーク
        self.w1_layer = nn.Linear(state_dim + z_dim, num_agents * hidden_dim)
        self.b1_layer = nn.Linear(state_dim + z_dim, hidden_dim)
        self.w2_layer = nn.Linear(state_dim + z_dim, hidden_dim)
        self.b2_layer = nn.Linear(state_dim + z_dim, 1)

    def forward(self, agent_qs, states, z):
        """
        agent_qs: (batch_size, num_agents)
        states: (batch_size, state_dim)
        z: (batch_size, z_dim) を想定
        """
        batch_size = agent_qs.size(0)

        # z が3次元の場合（(batch_size, num_agents, z_dim)）、エージェント次元を削除
        if z.dim() == 3:
            z = z[:, 0, :]  # エージェント0のzを使用（全エージェントで同じzを想定）

        # states と z を結合
        sz = torch.cat([states, z], dim=-1)  # (batch_size, state_dim + z_dim)

        # 第1層の重みとバイアスを生成
        w1 = self.w1_layer(sz).view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.b1_layer(sz).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # 第1層の計算
        agent_qs_expanded = agent_qs.unsqueeze(-1)  # (batch_size, num_agents, 1)
        h = F.elu(torch.bmm(agent_qs_expanded.transpose(1, 2), w1) + b1)  # (batch_size, 1, hidden_dim)

        # 第2層の重みとバイアスを生成
        w2 = self.w2_layer(sz).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        b2 = self.b2_layer(sz).unsqueeze(1)  # (batch_size, 1, 1)

        # 第2層の計算
        q_tot = torch.bmm(h, w2.transpose(1, 2)) + b2  # (batch_size, 1, 1)
        return q_tot.squeeze(-1).squeeze(-1)  # (batch_size,)

class VariationalEncoder(nn.Module):
    """潜在変数 z の変分分布 q_psi(z|s) を学習するエンコーダ"""
    def __init__(self, state_dim, z_dim, hidden_dim=64):
        super().__init__()
        self.fc_mu = nn.Linear(state_dim, z_dim)
        self.fc_logvar = nn.Linear(state_dim, z_dim)

    def forward(self, state):
        # state: (batch_size, state_dim)
        mu = self.fc_mu(state)
        logvar = self.fc_logvar(state)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    

class MAVENMemory:
    def __init__(self):
        self.obs = []          # 各エージェントの観測 (T, num_agents, obs_dim)
        self.actions = []      # 実行された行動 (T, num_agents)
        self.rewards = []      # 得られた報酬 (T, num_agents)
        self.next_obs = []     # 次の観測 (T, num_agents, obs_dim)
        self.dones = []        # 終了フラグ (T,)
        self.states = []       # グローバル状態 (T, state_dim)
        self.next_states = []  # 次のグローバル状態 (T, state_dim)
        self.z = []            # 潜在変数 z (T, z_dim)

    def store(self, obs, actions, rewards, next_obs, done, state, next_state, z):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_obs.append(next_obs)
        self.dones.append(done)
        self.states.append(state)
        self.next_states.append(next_state)
        self.z.append(z)

    def clear(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []
        self.states = []
        self.next_states = []
        self.z = []

    def get_batch(self):
        return {
            'obs': torch.stack(self.obs),
            'actions': torch.stack(self.actions),
            'rewards': torch.stack(self.rewards),
            'next_obs': torch.stack(self.next_obs),
            'dones': torch.tensor(self.dones, dtype=torch.float),
            'states': torch.stack(self.states),
            'next_states': torch.stack(self.next_states),
            'z': torch.stack(self.z),
        }
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """各エージェント用のQネットワーク（zを入力に追加）"""
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, z):
        # x: (batch_size, obs_dim), z: (batch_size, z_dim) を想定
        # z が3次元の場合（(batch_size, num_agents, z_dim)）、エージェント次元を削除
        if z.dim() == 3:
            z = z[:, 0, :]  # エージェント0のzを使用（全エージェントで同じzを想定）
        xz = torch.cat([x, z], dim=-1)
        xz = F.relu(self.fc1(xz))
        xz = F.relu(self.fc2(xz))
        q_values = self.fc3(xz)
        return q_values

class MixingNetwork(nn.Module):
    """QMIXのMixing Network（zを入力に追加）"""
    def __init__(self, num_agents, state_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        # 重みとバイアスを生成するネットワーク
        self.w1_layer = nn.Linear(state_dim + z_dim, num_agents * hidden_dim)
        self.b1_layer = nn.Linear(state_dim + z_dim, hidden_dim)
        self.w2_layer = nn.Linear(state_dim + z_dim, hidden_dim)
        self.b2_layer = nn.Linear(state_dim + z_dim, 1)

    def forward(self, agent_qs, states, z):
        """
        agent_qs: (batch_size, num_agents)
        states: (batch_size, state_dim)
        z: (batch_size, z_dim) を想定
        """
        batch_size = agent_qs.size(0)

        # z が3次元の場合（(batch_size, num_agents, z_dim)）、エージェント次元を削除
        if z.dim() == 3:
            z = z[:, 0, :]  # エージェント0のzを使用（全エージェントで同じzを想定）

        # states と z を結合
        sz = torch.cat([states, z], dim=-1)  # (batch_size, state_dim + z_dim)

        # 第1層の重みとバイアスを生成
        w1 = self.w1_layer(sz).view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.b1_layer(sz).unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # 第1層の計算
        agent_qs_expanded = agent_qs.unsqueeze(-1)  # (batch_size, num_agents, 1)
        h = F.elu(torch.bmm(agent_qs_expanded.transpose(1, 2), w1) + b1)  # (batch_size, 1, hidden_dim)

        # 第2層の重みとバイアスを生成
        w2 = self.w2_layer(sz).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        b2 = self.b2_layer(sz).unsqueeze(1)  # (batch_size, 1, 1)

        # 第2層の計算
        q_tot = torch.bmm(h, w2.transpose(1, 2)) + b2  # (batch_size, 1, 1)
        return q_tot.squeeze(-1).squeeze(-1)  # (batch_size,)

class VariationalEncoder(nn.Module):
    """変分エンコーダ：q_psi(z|s)"""
    def __init__(self, state_dim, z_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class MAVENMemory:
    """MAVEN用：観測、行動、報酬、状態、zを保存するメモリ"""
    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.done = []
        self.states = []
        self.next_states = []
        self.z = []

    def store(self, obs, actions, rewards, next_obs, done, state, next_state, z):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_obs.append(next_obs)
        self.done.append(done)
        self.states.append(state)
        self.next_states.append(next_state)
        self.z.append(z)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.obs), batch_size, replace=False)
        obs_batch = torch.stack([self.obs[i] for i in indices])
        actions_batch = torch.stack([self.actions[i] for i in indices])
        rewards_batch = torch.stack([self.rewards[i] for i in indices])
        next_obs_batch = torch.stack([self.next_obs[i] for i in indices])
        done_batch = torch.tensor([self.done[i] for i in indices], dtype=torch.bool)
        states_batch = torch.stack([self.states[i] for i in indices])
        next_states_batch = torch.stack([self.next_states[i] for i in indices])
        z_batch = torch.stack([self.z[i] for i in indices])
        return obs_batch, actions_batch, rewards_batch, next_obs_batch, done_batch, states_batch, next_states_batch, z_batch

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_obs.clear()
        self.done.clear()
        self.states.clear()
        self.next_states.clear()
        self.z.clear()

class MAVENTrainer:
    """MAVEN用：QMIX + VAE の学習器"""
    def __init__(self, obs_dim, action_dim, num_agents=2, state_dim=None, z_dim=4,
                 gamma=0.95, lr=1e-3, beta=0.01, batch_size=32, tau=0.01):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.gamma = gamma
        self.beta = beta  # KLダイバージェンスの係数
        self.batch_size = batch_size  # バッチサイズ
        self.tau = tau  # ターゲットネットワークの更新係数

        if state_dim is None:
            state_dim = obs_dim * num_agents

        # 各エージェントのQネットワーク（zを入力）
        self.q_nets = nn.ModuleList([
            QNetwork(obs_dim, action_dim, z_dim).to(device) for _ in range(num_agents)
        ])
        self.target_q_nets = nn.ModuleList([
            QNetwork(obs_dim, action_dim, z_dim).to(device) for _ in range(num_agents)
        ])
        for target_q in self.target_q_nets:
            target_q.load_state_dict(self.q_nets[0].state_dict())

        # Mixing Network（zを入力）
        self.mixing_net = MixingNetwork(num_agents, state_dim, z_dim).to(device)
        self.target_mixing_net = MixingNetwork(num_agents, state_dim, z_dim).to(device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # 変分エンコーダ（q_psi(z|s)）
        self.variational_encoder = VariationalEncoder(state_dim, z_dim).to(device)

        # オプティマイザ（QMIX + VAE をまとめて最適化）
        all_params = (
            list(self.q_nets.parameters()) +
            list(self.mixing_net.parameters()) +
            list(self.variational_encoder.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr)

        # 温度パラメータ（ボルツマン探索用）
        self.temperature = 1.0
        self.min_temperature = 0.1
        self.temp_decay = 0.995

    def normalize_obs(self, obs_list):
        """観測をテンソルに変換"""
        return torch.FloatTensor(np.array(obs_list)).to(device)

    def train(self, memory):
        """
        MAVEN用：QMIX損失とVAE損失を同時に最適化
        """
        if len(memory.obs) < self.batch_size:
            return 0.0, 0.0

        # バッチ取得
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done, batch_states, batch_next_states, batch_z = memory.sample(self.batch_size)

        # z の形状を (batch_size, z_dim) に統一
        if batch_z.dim() == 3:
            batch_z = batch_z[:, 0, :]  # (batch_size, z_dim)

        # QMIX損失の計算
        agent_q_values = []
        agent_next_q_values = []
        for i in range(self.num_agents):
            # 現在のQ値（z を渡す）
            q_i = self.q_nets[i](batch_obs[:, i], batch_z)  # (batch_size, action_dim)
            agent_q_values.append(q_i.gather(1, batch_actions[:, i].unsqueeze(-1)).squeeze(-1))

            # ターゲットQ値（z を渡す）
            with torch.no_grad():
                next_q_i = self.target_q_nets[i](batch_next_obs[:, i], batch_z)  # (batch_size, action_dim)
                agent_next_q_values.append(next_q_i.max(dim=-1)[0])

        agent_q_values = torch.stack(agent_q_values, dim=1)  # (batch_size, num_agents)
        agent_next_q_values = torch.stack(agent_next_q_values, dim=1)  # (batch_size, num_agents)

        # Mixing NetworkでグローバルQ値を計算（z を渡す）
        q_tot = self.mixing_net(agent_q_values, batch_states, batch_z)
        next_q_tot = self.target_mixing_net(agent_next_q_values, batch_next_states, batch_z)

        # TDターゲット
        targets = batch_rewards.sum(dim=1) + self.gamma * (1 - batch_done.float()) * next_q_tot

        # QMIX損失
        loss_qmix = F.mse_loss(q_tot, targets.detach())

        # VAE損失の計算（z の分布と事前分布のKLダイバージェンス）
        mu, logvar = self.variational_encoder(batch_states)
        loss_vae = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / self.batch_size

        # 合計損失
        total_loss = loss_qmix + self.beta * loss_vae

        # 勾配更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # ターゲットネットワークのソフト更新
        self.soft_update_module_list(self.target_q_nets, self.q_nets, self.tau)
        self.soft_update_module(self.target_mixing_net, self.mixing_net, self.tau)

        return loss_qmix.item(), loss_vae.item()

    def soft_update_module_list(self, target_nets, nets, tau):
        """ターゲットネットワークのソフト更新（ModuleList用）"""
        for target_net, net in zip(target_nets, nets):
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def soft_update_module(self, target_net, net, tau):
        """ターゲットネットワークのソフト更新（単一Module用）"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def select_action(self, obs_tensor, i, z, training=True):
        """
        MAVEN用：z に依存した行動選択（学習時はsoftmax、評価時はargmax）
        """
        with torch.no_grad():
            # z が3次元の場合（(1, num_agents, z_dim)）、エージェント次元を削除
            if z.dim() == 3:
                z = z[:, 0, :]  # (1, z_dim)
            # QNetworkに obs_tensor[i] と z を渡す
            q_values = self.q_nets[i](obs_tensor[i].unsqueeze(0), z)  # (1, action_dim)
            if training:
                probs = F.softmax(q_values, dim=-1)
                action = torch.multinomial(probs, 1).item()
            else:
                action = q_values.argmax().item()
        return action

    def sample_z(self, state):
        """状態から z をサンプリング（エピソード開始時に呼ぶ）"""
        with torch.no_grad():
            mu, logvar = self.variational_encoder(state.unsqueeze(0))
            z = self.variational_encoder.reparameterize(mu, logvar)
        return z.squeeze(0)

    def update_temperature(self):
        """温度パラメータの更新（エピソードごとに呼ぶ）"""
        self.temperature = max(self.min_temperature, self.temperature * self.temp_decay)