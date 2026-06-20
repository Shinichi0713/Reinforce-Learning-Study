import numpy as np
from collections import defaultdict

class MultiAgentBuffer:
    def __init__(self, num_agents, obs_dim, state_dim, action_dim):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'global_states': [],
            'log_probs': [],
            'values': [],       # 各エージェントの辞書形式 {'pursuer_0': v, ...}
            'terminated': [],
            'truncated': [],
            'advantages': [],   # compute_advantages内で辞書のリストとして生成
            'returns': [],      # compute_advantages内で辞書のリストとして生成
        }

        self.episode_lengths = []
        self.current_episode_steps = 0

    def store(self, obs_dict, action_dict, reward_dict, global_state,
              log_prob_dict, value_dict, terminated, truncated):
        """
        環境の1ステップのデータを保存
        """
        self.buffer['obs'].append(obs_dict)
        self.buffer['actions'].append(action_dict)
        self.buffer['rewards'].append(reward_dict)
        self.buffer['global_states'].append(global_state)
        self.buffer['log_probs'].append(log_prob_dict)
        self.buffer['values'].append(value_dict)
        self.buffer['terminated'].append(terminated)
        self.buffer['truncated'].append(truncated)

        self.current_episode_steps += 1

        # エピソードが終了（または時間切れ）したら長さを記録
        if terminated or truncated:
            self.episode_lengths.append(self.current_episode_steps)
            self.current_episode_steps = 0

    def sample(self, batch_size):
        if len(self.buffer['obs']) == 0:
            return None

        # 実際に計算が完了しているデータ数を上限にする（安全策）
        total_stored = len(self.buffer['obs'])
        indices = np.random.choice(total_stored, size=batch_size, replace=False)

        obs_batch, actions_batch, rewards_batch = [], [], []
        log_probs_batch, advantages_batch, returns_batch = [], [], []
        global_states_batch, values_batch = [], []

        for idx in indices:
            obs_step, actions_step, rewards_step = [], [], []
            log_probs_step, advantages_step, returns_step = [], [], []
            values_step = []

            for i in range(self.num_agents):
                agent_name = f'pursuer_{i}'
                obs_step.append(self.buffer['obs'][idx][agent_name])
                actions_step.append(self.buffer['actions'][idx][agent_name])
                log_probs_step.append(self.buffer['log_probs'][idx][agent_name])
                values_step.append(self.buffer['values'][idx][agent_name])

                # MAPPO.update側が「rewards」というキーでCriticのターゲット（Returns）を要求するため、
                # ここで returns の値を rewards_step にマッピングします。
                if idx < len(self.buffer['returns']) and self.buffer['returns'][idx] is not None and agent_name in self.buffer['returns'][idx]:
                    rewards_step.append(self.buffer['returns'][idx][agent_name])
                else:
                    rewards_step.append(0.0)

                # 安全に個別advantageを取得
                if idx < len(self.buffer['advantages']) and self.buffer['advantages'][idx] is not None and agent_name in self.buffer['advantages'][idx]:
                    advantages_step.append(self.buffer['advantages'][idx][agent_name])
                else:
                    advantages_step.append(0.0)

            obs_batch.append(obs_step)
            actions_batch.append(actions_step)
            rewards_batch.append(rewards_step)
            log_probs_batch.append(log_probs_step)
            advantages_batch.append(advantages_step)
            values_batch.append(values_step)

            # global_statesの保存 (batch, state_dim)
            global_states_batch.append(self.buffer['global_states'][idx])

        batch = {
            'obs': np.array(obs_batch, dtype=np.float32),                 # (batch, num_agents, obs_dim)
            'actions': np.array(actions_batch, dtype=np.int64),           # (batch, num_agents)
            'rewards': np.array(rewards_batch, dtype=np.float32),         # (batch, num_agents) ※実質は個別Returns
            'global_states': np.array(global_states_batch, dtype=np.float32), # (batch, state_dim)
            'log_probs': np.array(log_probs_batch, dtype=np.float32),     # (batch, num_agents)
            'values': np.array(values_batch, dtype=np.float32),           # (batch, num_agents)
            'advantages': np.array(advantages_batch, dtype=np.float32),   # (batch, num_agents)
        }
        return batch

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        total_steps = len(self.buffer['obs'])
        advantages = [None] * total_steps
        returns = [None] * total_steps

        start_idx = 0
        for ep_len in self.episode_lengths:
            end_idx = start_idx + ep_len

            # 該当エピソードのデータをスライス
            rewards_ep = self.buffer['rewards'][start_idx:end_idx]
            values_ep = self.buffer['values'][start_idx:end_idx]
            terminated_ep = self.buffer['terminated'][start_idx:end_idx]
            truncated_ep = self.buffer['truncated'][start_idx:end_idx]

            for agent_idx in range(self.num_agents):
                agent_name = f'pursuer_{agent_idx}'
                agent_rewards = [r[agent_name] for r in rewards_ep]
                agent_values = [v[agent_name] for v in values_ep]

                advantage_ep = np.zeros(ep_len)
                return_ep = np.zeros(ep_len)

                gae = 0.0

                # 🌟 時間切れ（truncated）発生時のブートストラップ初期値設定の厳密化
                # エピソードの最後のステップの「次の状態の価値 V(s_{t+1})」を決定します
                if truncated_ep[-1]:
                    # 時間切れの場合は、途中で遮断されただけなので最後の状態の価値をブートストラップとして使う
                    next_value = agent_values[-1]
                    future_return = agent_values[-1]
                else:
                    # 通常終了（terminated）またはその他の場合は 0.0
                    next_value = 0.0
                    future_return = 0.0

                # 未来(お尻)から過去(頭)へ向かって逆順ループ
                for t in reversed(range(ep_len)):
                    # 🌟 次のステップ t+1 が存在するか、あるいはエピソード終端かでマスクを設定
                    # 通常、PPOでは「現在のステップ t が終了フラグ（terminated）を持つ場合、次の価値を0」とマスクします
                    is_not_terminal = 0.0 if terminated_ep[t] else 1.0

                    # TD誤差 𝛿_t = r_t + 𝛾 * V(s_{t+1}) * mask - V(s_t)
                    delta = agent_rewards[t] + gamma * next_value * is_not_terminal - agent_values[t]

                    # GAEの累積: 𝛿_t + 𝛾 * 𝜆 * GAE_{t+1} * mask
                    gae = delta + gamma * gae_lambda * is_not_terminal * gae
                    advantage_ep[t] = gae

                    # 割引累積報酬（Return）の計算
                    future_return = agent_rewards[t] + gamma * is_not_terminal * future_return
                    return_ep[t] = future_return

                    # 1つ過去のステップ（t-1）にとっての「未来の価値 V(s_{t+1})」は、現在の V(s_t)
                    next_value = agent_values[t]

                # 各エージェントの計算結果をステップごとの辞書に戻して全体へ格納
                for t_idx in range(ep_len):
                    global_idx = start_idx + t_idx
                    if advantages[global_idx] is None:
                        advantages[global_idx] = {}
                        returns[global_idx] = {}
                    advantages[global_idx][agent_name] = advantage_ep[t_idx]
                    returns[global_idx][agent_name] = return_ep[t_idx]

            start_idx = end_idx

        self.buffer['advantages'] = advantages
        self.buffer['returns'] = returns

    def clear(self):
        for key in self.buffer:
            self.buffer[key] = []
        self.episode_lengths = []
        self.current_episode_steps = 0

    def __len__(self):
        return len(self.buffer['obs'])