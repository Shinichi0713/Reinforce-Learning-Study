import torch
import torch.nn as nn
import torch.nn.functional as F

class QMixAgent:
    def __init__(self, n_agents, obs_shape, state_shape, n_actions, ...):
        # 1. ネットワークの初期化
        self.agent_net = RNNAgent(obs_shape, hidden_dim=64, n_actions=n_actions)
        self.mixer_net = QMixer(n_agents, state_shape, mixing_embed_dim=32, hypernet_embed_dim=64)
        
        # 2. ターゲットネットワークの初期化 (安定化のため)
        self.target_agent_net = RNNAgent(obs_shape, hidden_dim=64, n_actions=n_actions)
        self.target_mixer_net = QMixer(n_agents, state_shape, mixing_embed_dim=32, hypernet_embed_dim=64)
        
        # ターゲットネットワークの重みをコピー
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())
        
        # 3. オプティマイザの定義
        params = list(self.agent_net.parameters()) + list(self.mixer_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=5e-4)

    def train(self, batch):
        # バッチからデータを取り出し、形状を調整
        # obs_batch, state_batch, action_batch, reward_batch, next_obs_batch, next_state_batch, ...
        
        # 1. 現在のQ値 (Q_tot) の計算
        # 各エージェントのQ値を計算
        agent_qs = []
        for i in range(self.n_agents):
            # 隠れ状態 h はタイムステップごとに更新される
            # 実際の実装では、バッチ処理でシーケンス全体を処理する
            q, _ = self.agent_net(batch['obs'][:, i], initial_hidden_state)
            # 実行されたアクションのQ値を選択
            chosen_q = torch.gather(q, dim=-1, index=batch['action'][:, i].long())
            agent_qs.append(chosen_q)

        # 全エージェントのQ値を結合し、Mixing Networkに入力
        agent_qs = torch.cat(agent_qs, dim=-1)
        q_tot = self.mixer_net(agent_qs, batch['state'])

        # 2. TDターゲット (y_target) の計算
        # 次のステップでの最大のQ値 (Target Q_tot) を計算
        target_agent_qs = []
        for i in range(self.n_agents):
            # Target Agent Networkを使用
            target_q, _ = self.target_agent_net(batch['next_obs'][:, i], next_hidden_state)
            # 最大Q値を選択 (単調性により、個別のQ値の最大値を選択すれば良い)
            target_max_q = target_q.max(dim=-1)[0]
            target_agent_qs.append(target_max_q)
            
        target_agent_qs = torch.cat(target_agent_qs, dim=-1)
        target_q_tot = self.target_mixer_net(target_agent_qs, batch['next_state']).detach()

        # ターゲット値の完成: R + gamma * max Q_next
        td_target = batch['reward'] + gamma * target_q_tot * (1 - batch['terminated'])

        # 3. 損失の計算と最適化
        loss = F.mse_loss(q_tot, td_target.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 4. ターゲットネットワークの更新 (ソフト/ハードアップデート)
        if update_counter % target_update_interval == 0:
            self.target_agent_net.load_state_dict(self.agent_net.state_dict())
            self.target_mixer_net.load_state_dict(self.mixer_net.state_dict())

