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
            'values': [],
            'terminated': [],
            'truncated': [],
            'advantages': [], # 予め初期化
            'returns': [],    # 予め初期化
        }

        self.episode_lengths = []
        self.current_episode_steps = 0

    def store(self, obs_dict, action_dict, reward_dict, global_state,
              log_prob_dict, value, terminated, truncated):
        self.buffer['obs'].append(obs_dict)
        self.buffer['actions'].append(action_dict)
        self.buffer['rewards'].append(reward_dict)
        self.buffer['global_states'].append(global_state)
        self.buffer['log_probs'].append(log_prob_dict)
        self.buffer['values'].append(value)
        self.buffer['terminated'].append(terminated)
        self.buffer['truncated'].append(truncated)

        self.current_episode_steps += 1

        if terminated or truncated:
            self.episode_lengths.append(self.current_episode_steps)
            self.current_episode_steps = 0

    def sample(self, batch_size):
        if len(self.buffer['obs']) == 0:
            return None

        indices = np.random.choice(len(self.buffer['obs']), size=batch_size, replace=False)

        obs_batch, actions_batch, rewards_batch = [], [], []
        log_probs_batch, advantages_batch, returns_batch = [], [], []
        global_states_batch, values_batch = [], []

        for idx in indices:
            obs_step, actions_step, rewards_step = [], [], []
            log_probs_step, advantages_step, returns_step = [], [], []

            for i in range(self.num_agents):
                agent_name = f'pursuer_{i}'
                obs_step.append(self.buffer['obs'][idx][agent_name])
                actions_step.append(self.buffer['actions'][idx][agent_name])
                rewards_step.append(self.buffer['rewards'][idx][agent_name])
                log_probs_step.append(self.buffer['log_probs'][idx][agent_name])
                
                # 安全にadvantageとreturnを取得
                if idx < len(self.buffer['advantages']):
                    advantages_step.append(self.buffer['advantages'][idx][agent_name])
                    returns_step.append(self.buffer['returns'][idx][agent_name])
                else:
                    advantages_step.append(0.0)
                    returns_step.append(0.0)

            obs_batch.append(obs_step)
            actions_batch.append(actions_step)
            rewards_batch.append(rewards_step)
            log_probs_batch.append(log_probs_step)
            advantages_batch.append(advantages_step)
            returns_batch.append(returns_step)

            global_states_batch.append(self.buffer['global_states'][idx])
            values_batch.append(self.buffer['values'][idx])

        batch = {
            'obs': np.array(obs_batch, dtype=np.float32),
            'actions': np.array(actions_batch, dtype=np.int64),
            'rewards': np.array(returns_batch, dtype=np.float32), # ⚠️ MAPPO.update側が「rewards」というキーでリターンを受け取っているため、ここに returns_batch をマッピングします
            'global_states': np.array(global_states_batch, dtype=np.float32),
            'log_probs': np.array(log_probs_batch, dtype=np.float32),
            'values': np.array(values_batch, dtype=np.float32),
            'advantages': np.array(advantages_batch, dtype=np.float32),
        }
        return batch

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        advantages = [None] * len(self.buffer['obs'])
        returns = [None] * len(self.buffer['obs'])

        start_idx = 0
        for ep_len in self.episode_lengths:
            end_idx = start_idx + ep_len

            rewards_ep = [self.buffer['rewards'][i] for i in range(start_idx, end_idx)]
            values_ep = [self.buffer['values'][i] for i in range(start_idx, end_idx)]
            terminated_ep = [self.buffer['terminated'][i] for i in range(start_idx, end_idx)]
            truncated_ep = [self.buffer['truncated'][i] for i in range(start_idx, end_idx)]

            for agent_idx in range(self.num_agents):
                agent_name = f'pursuer_{agent_idx}'
                agent_rewards = [r[agent_name] for r in rewards_ep]
                agent_values = values_ep.copy()

                # 【改善】未来から過去へ向かって正しくループを回す
                advantage_ep = np.zeros(ep_len)
                return_ep = np.zeros(ep_len)
                
                gae = 0.0
                future_return = 0.0
                
                # エピソードの最後の次の状態の価値（初期値）
                # 本当は環境の最終ステップの次の状態の価値が必要ですが、
                # 終了時(terminated)なら0、時間切れ(truncated)なら最後の価値でブートストラップします
                if truncated_ep[-1]:
                    next_value = agent_values[-1]
                    future_return = agent_values[-1]
                else:
                    next_value = 0.0
                    future_return = 0.0

                for t in reversed(range(ep_len)):
                    # TD誤差 𝛿_t = r_t + 𝛾 * V(s_{t+1}) - V(s_t)
                    delta = agent_rewards[t] + gamma * next_value - agent_values[t]
                    
                    # GAEの累積
                    gae = delta + gamma * gae_lambda * (0.0 if terminated_ep[t] else gae)
                    advantage_ep[t] = gae
                    
                    # 割引累積報酬（Return）の計算
                    future_return = agent_rewards[t] + gamma * (0.0 if terminated_ep[t] else future_return)
                    return_ep[t] = future_return
                    
                    # 次のループ（過去ステージ）にとって、現在の価値が「未来の価値(next_value)」になる
                    next_value = agent_values[t]

                # 計算した値をステップごとの辞書構造に戻して全体リストへ格納
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