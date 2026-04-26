import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# 環境（あなたのコード）
# from your_env_file import DroneDeliveryEnv

# ハイパーパラメータ（例）
BATCH_SIZE = 32
BUFFER_SIZE = 10000
GAMMA = 0.99
LR = 1e-3
RND_BETA = 0.1  # 内部報酬の重み
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 100


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 状態: agent_pos(2), carrying(1), packages情報, other_agent_pos(2) などをフラット化して入力
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class MixerNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim=64):
        super().__init__()
        self.num_agents = num_agents
        self.hyper_w1 = nn.Linear(state_dim, num_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, agent_qs, state):
        # agent_qs: [batch, num_agents]
        # state: [batch, state_dim]
        bs = agent_qs.size(0)

        # 第1層
        w1 = torch.abs(self.hyper_w1(state))  # [batch, num_agents * hidden]
        w1 = w1.view(bs, self.num_agents, -1)  # [batch, num_agents, hidden]
        b1 = self.hyper_b1(state).unsqueeze(1)  # [batch, 1, hidden]

        agent_qs = agent_qs.unsqueeze(-1)  # [batch, num_agents, 1]
        h = torch.relu(torch.bmm(agent_qs.transpose(1, 2), w1) + b1)  # [batch, 1, hidden]

        # 第2層
        w2 = torch.abs(self.hyper_w2(state))  # [batch, hidden]
        w2 = w2.unsqueeze(-1)  # [batch, hidden, 1]  ← ここを修正
        b2 = self.hyper_b2(state)  # [batch, 1]

        # h: [batch, 1, hidden]
        # w2: [batch, hidden, 1]
        q_total = torch.bmm(h, w2) + b2.unsqueeze(-1)  # [batch, 1, 1]
        return q_total.squeeze(-1).squeeze(-1)  # [batch]

class RNDNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
  

class QMIXRNDAgent:
    def __init__(self, env, state_dim, action_dim, device="cpu"):
        print(f"state_dim = {state_dim}, type = {type(state_dim)}")
        self.env = env
        self.num_agents = env.num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Q ネットワーク（各エージェント）
        self.q_nets = [QNetwork(state_dim, action_dim).to(device) for _ in range(self.num_agents)]
        self.target_q_nets = [QNetwork(state_dim, action_dim).to(device) for _ in range(self.num_agents)]
        for tq in self.target_q_nets:
            tq.load_state_dict(self.q_nets[0].state_dict())  # 初期は同じ重み

        # Mixer ネットワーク
        # グローバル状態の次元は適宜定義（例: 全エージェント位置 + 全パッケージ状態）
        global_state_dim = state_dim * self.num_agents
        self.mixer = MixerNetwork(self.num_agents, global_state_dim).to(device)
        self.target_mixer = MixerNetwork(self.num_agents, global_state_dim).to(device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # RND ネットワーク
        self.rnd_target = RNDNetwork(state_dim).to(device)
        self.rnd_predict = RNDNetwork(state_dim).to(device)

        # オプティマイザ
        self.q_optimizers = [optim.Adam(qnet.parameters(), lr=LR) for qnet in self.q_nets]
        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=LR)
        self.rnd_optimizer = optim.Adam(self.rnd_predict.parameters(), lr=LR)

        # リプレイバッファ
        self.buffer = deque(maxlen=BUFFER_SIZE)

        # 探索率
        self.eps = EPS_START

        print("qmix agent is initialized")

    def _obs_to_tensor(self, obs_list):
        states = []
        for obs in obs_list:
            vec = []

            # agent_pos
            vec.extend(obs["agent_pos"])

            # carrying
            vec.append(obs["carrying"])

            # other_agent_pos
            vec.extend(obs["other_agent"])

            # packages
            for pack in obs["packages"]:
                pick, drop, picked, delivered = pack
                vec.extend(pick)   # pick_pos
                vec.extend(drop)   # drop_pos
                vec.append(1.0 if picked else 0.0)
                vec.append(1.0 if delivered else 0.0)

            states.append(np.array(vec, dtype=np.float32))

        return torch.FloatTensor(np.array(states)).to(self.device)

    def _global_state_to_tensor(self, obs_list):
        global_vec = []
        for obs in obs_list:
            # 各エージェントの状態ベクトルをそのまま結合
            vec = []
            vec.extend(obs["agent_pos"])
            vec.append(obs["carrying"])
            vec.extend(obs["other_agent"])
            for pack in obs["packages"]:
                pick, drop, picked, delivered = pack
                vec.extend(pick)
                vec.extend(drop)
                vec.append(1.0 if picked else 0.0)
                vec.append(1.0 if delivered else 0.0)
            global_vec.extend(vec)

        return torch.FloatTensor(global_vec).unsqueeze(0).to(self.device)

    def act(self, obs_list, explore=True):
        # obs_list: list[dict] -> 各エージェントの行動を返す
        actions = []
        state_tensor = self._obs_to_tensor(obs_list)  # [num_agents, state_dim]

        for i in range(self.num_agents):
            if explore and random.random() < self.eps:
                # ε-greedy
                a = random.randint(0, self.action_dim - 1)
            else:
                with torch.no_grad():
                    q_vals = self.q_nets[i](state_tensor[i].unsqueeze(0))  # [1, action_dim]
                    a = q_vals.argmax().item()
            actions.append(a)

        return actions

    def compute_intrinsic_reward(self, obs_list):
        # RND による内部報酬の計算
        state_tensor = self._obs_to_tensor(obs_list)  # [num_agents, state_dim]
        intrinsic_rewards = []

        for i in range(self.num_agents):
            s = state_tensor[i].unsqueeze(0)
            with torch.no_grad():
                target_feat = self.rnd_target(s)
                pred_feat = self.rnd_predict(s)
                error = ((target_feat - pred_feat) ** 2).mean().item()
            intrinsic_rewards.append(RND_BETA * error)

        return intrinsic_rewards

    def update_rnd(self, obs_list):
        # RND 予測ネットワークの学習
        state_tensor = self._obs_to_tensor(obs_list)  # [num_agents, state_dim]
        loss_sum = 0

        for i in range(self.num_agents):
            s = state_tensor[i].unsqueeze(0)
            target_feat = self.rnd_target(s).detach()
            pred_feat = self.rnd_predict(s)
            loss = ((target_feat - pred_feat) ** 2).mean()
            loss_sum += loss

        self.rnd_optimizer.zero_grad()
        loss_sum.backward()
        self.rnd_optimizer.step()

    def store_transition(self, obs, actions, rewards, next_obs, done):
        # 簡略化のため、全エージェント分をまとめて保存
        transition = (obs, actions, rewards, next_obs, done)
        self.buffer.append(transition)

    def update_qmix(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        batch = random.sample(self.buffer, BATCH_SIZE)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)

        for i in range(self.num_agents):
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            global_states = []
            next_global_states = []

            for j in range(BATCH_SIZE):
                obs_list = obs_batch[j]
                next_obs_list = next_obs_batch[j]
                states.append(self._obs_to_tensor(obs_list)[i])
                actions.append(act_batch[j][i])  # エージェント i の行動
                rewards.append(rew_batch[j][i])
                next_states.append(self._obs_to_tensor(next_obs_list)[i])
                dones.append(done_batch[j])
                global_states.append(self._global_state_to_tensor(obs_list).squeeze(0))
                next_global_states.append(self._global_state_to_tensor(next_obs_list).squeeze(0))

            states = torch.stack(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.stack(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)
            global_states = torch.stack(global_states).to(self.device)
            next_global_states = torch.stack(next_global_states).to(self.device)

            curr_q = self.q_nets[i](states)
            curr_q_a = curr_q.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_vals = self.target_q_nets[i](next_states)
                next_actions = next_q_vals.argmax(dim=1)

                next_agent_qs = []
                for k in range(self.num_agents):
                    if k == i:
                        next_agent_qs.append(next_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1))
                    else:
                        other_next_q = self.target_q_nets[k](next_states)
                        other_next_a = other_next_q.argmax(dim=1)
                        next_agent_qs.append(other_next_q.gather(1, other_next_a.unsqueeze(1)).squeeze(1))

                next_agent_qs = torch.stack(next_agent_qs, dim=1)
                next_q_total = self.target_mixer(next_agent_qs, next_global_states)
                target_q = rewards + (GAMMA * next_q_total * (~dones))

            curr_agent_qs = []
            for k in range(self.num_agents):
                if k == i:
                    curr_agent_qs.append(curr_q_a)
                else:
                    # 他エージェントの curr_q と行動を取得
                    other_curr_q = self.q_nets[k](states)
                    # ここが修正ポイント: act_batch[j][k] を使う
                    other_curr_a = [act_batch[j][k] for j in range(BATCH_SIZE)]  # バッチ内の全サンプルについてエージェント k の行動
                    other_curr_a = torch.LongTensor(other_curr_a).to(self.device)
                    other_curr_q_a = other_curr_q.gather(1, other_curr_a.unsqueeze(1)).squeeze(1)
                    curr_agent_qs.append(other_curr_q_a)

            curr_agent_qs = torch.stack(curr_agent_qs, dim=1)
            curr_q_total = self.mixer(curr_agent_qs, global_states)

            loss = ((curr_q_total - target_q.detach()) ** 2).mean()

            self.q_optimizers[i].zero_grad()
            self.mixer_optimizer.zero_grad()
            loss.backward()
            self.q_optimizers[i].step()
            self.mixer_optimizer.step()

        self.eps = max(EPS_END, self.eps * EPS_DECAY)

    def update_targets(self):
        for i in range(self.num_agents):
            self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())