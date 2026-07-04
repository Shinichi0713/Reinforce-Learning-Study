import numpy as np
from collections import defaultdict

class MultiAgentBuffer:
    def __init__(self, num_agents, obs_dim, state_dim, action_dim):
        self.num_agents = num_agents
        self.obs_dim = obs_dim          # 🌟 236 (196空間 + 40行動履歴)
        self.state_dim = state_dim      # 🌟 1888 (236 x 8)
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

        total_stored = len(self.buffer['obs'])
        indices = np.random.choice(total_stored, size=batch_size, replace=False)

        obs_batch, actions_batch, rewards_batch = [], [], []
        log_probs_batch, advantages_batch, returns_batch = [], [], []
        values_batch, global_states_batch = []

        for idx in indices:
            obs_step, actions_step, log_probs_step = [], [], []
            values_step, advantages_step, returns_step = [], [], []

            for i in range(self.num_agents):
                agent_name = f'pursuer_{i}'
                obs_step.append(self.buffer['obs'][idx][agent_name])
                actions_step.append(self.buffer['actions'][idx][agent_name])
                log_probs_step.append(self.buffer['log_probs'][idx][agent_name])
                values_step.append(self.buffer['values'][idx][agent_name])

                # 🌟 変更点1: 報酬（Returns）バッチのサンプリングキーを正しく設定
                # 元のコードの rewards_step.append(self.buffer['returns']...) に合わせつつ安全に取得
                if idx < len(self.buffer['returns']) and self.buffer['returns'][idx] is not None and agent_name in self.buffer['returns'][idx]:
                    returns_step.append(self.buffer['returns'][idx][agent_name])
                else:
                    returns_step.append(0.0)

                # 🌟 変更点2: Advantageバッチのサンプリング
                if idx < len(self.buffer['advantages']) and self.buffer['advantages'][idx] is not None and agent_name in self.buffer['advantages'][idx]:
                    advantages_step.append(self.buffer['advantages'][idx][agent_name])
                else:
                    advantages_step.append(0.0)

            obs_batch.append(obs_step)
            actions_batch.append(actions_step)
            log_probs_batch.append(log_probs_step)
            values_batch.append(values_step)
            
            # 🌟 変更点3: リスト名とバッチキーの不整合を防ぐため、変数名をわかりやすく整理
            returns_batch.append(returns_step)
            advantages_batch.append(advantages_step)

            global_states_batch.append(self.buffer['global_states'][idx])

        # MAPPO.update() が期待するNumPy配列の形状に成形
        batch = {
            'obs': np.array(obs_batch, dtype=np.float32),                 # (batch, num_agents, 236)
            'actions': np.array(actions_batch, dtype=np.int64),           # (batch, num_agents)
            'rewards': np.array(returns_batch, dtype=np.float32),         # (batch, num_agents) ※MAPPO側で returns として受け取るもの
            'global_states': np.array(global_states_batch, dtype=np.float32), # (batch, num_agents, 236) またはフラット
            'log_probs': np.array(log_probs_batch, dtype=np.float32),     # (batch, num_agents)
            'values': np.array(values_batch, dtype=np.float32),           # (batch, num_agents)
            'advantages': np.array(advantages_batch, dtype=np.float32),   # (batch, num_agents)
        }
        return batch

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        total_steps = len(self.buffer['obs'])
        advantages = [None] * total_steps
        returns = [None] * total_steps

        # ロールアウト終了時に、まだ途中で終わっていない未完のエピソードの長さを一時的に追加
        working_lengths = list(self.episode_lengths)
        if self.current_episode_steps > 0:
            working_lengths.append(self.current_episode_steps)

        start_idx = 0
        for ep_len in working_lengths:
            end_idx = start_idx + ep_len

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

                # 時間切れ（truncated）またはロールアウトの強制的な終端である場合、最後の状態の価値でブートストラップ
                if truncated_ep[-1] or (not terminated_ep[-1] and start_idx + ep_len == total_steps):
                    next_value = agent_values[-1]
                    future_return = agent_values[-1]
                else:
                    next_value = 0.0
                    future_return = 0.0

                # 未来から過去へ向かって逆順ループ
                for t in reversed(range(ep_len)):
                    is_not_terminal = 0.0 if terminated_ep[t] else 1.0

                    delta = agent_rewards[t] + gamma * next_value * is_not_terminal - agent_values[t]
                    gae = delta + gamma * gae_lambda * is_not_terminal * gae
                    advantage_ep[t] = gae

                    future_return = agent_rewards[t] + gamma * is_not_terminal * future_return
                    return_ep[t] = future_return

                    next_value = agent_values[t]

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