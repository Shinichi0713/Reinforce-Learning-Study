import numpy as np
from collections import defaultdict

class MultiAgentBuffer:
    """
    MAPPO用のマルチエージェントメモリバッファ
    
    特徴:
    - ステップ単位で全エージェント分の経験を保存
    - リスト／辞書ベースのシンプルな構造
    - ミニバッチサンプリング（ランダム）
    - Advantageは後計算（GAEなどで別途計算）
    """
    
    def __init__(self, num_agents, obs_dim, state_dim, action_dim):
        """
        Args:
            num_agents: エージェント数（Pursuitなら8）
            obs_dim: 各エージェントの観測次元（147）
            state_dim: グローバル状態の次元（147*8=1176）
            action_dim: 行動空間のサイズ（Pursuitの行動数）
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # バッファ（辞書ベース）
        self.buffer = {
            'obs': [],           # 各エージェントの観測 [{'pursuer_0': obs0, ...}, ...]
            'actions': [],       # 各エージェントの行動 [{'pursuer_0': a0, ...}, ...]
            'rewards': [],       # 各エージェントの報酬 [{'pursuer_0': r0, ...}, ...]
            'global_states': [], # グローバル状態 [state, ...]
            'log_probs': [],     # 行動の対数確率 [{'pursuer_0': log_p0, ...}, ...]
            'values': [],        # Criticの価値推定 [value, ...]
            'terminated': [],    # 終了フラグ [bool, ...]
            'truncated': [],     # 打ち切りフラグ [bool, ...]
        }
        
        # エピソードの長さを記録（後でAdvantage計算に使う）
        self.episode_lengths = []
        self.current_episode_steps = 0

    def store(self, obs_dict, action_dict, reward_dict, global_state, 
              log_prob_dict, value, terminated, truncated):
        """
        1ステップ分の経験を保存
        
        Args:
            obs_dict: {'pursuer_0': obs0, 'pursuer_1': obs1, ...}
            action_dict: {'pursuer_0': a0, 'pursuer_1': a1, ...}
            reward_dict: {'pursuer_0': r0, 'pursuer_1': r1, ...}
            global_state: グローバル状態（1176次元）
            log_prob_dict: {'pursuer_0': log_p0, 'pursuer_1': log_p1, ...}
            value: Criticの価値推定（スカラー）
            terminated: 終了フラグ
            truncated: 打ち切りフラグ
        """
        self.buffer['obs'].append(obs_dict)
        self.buffer['actions'].append(action_dict)
        self.buffer['rewards'].append(reward_dict)
        self.buffer['global_states'].append(global_state)
        self.buffer['log_probs'].append(log_prob_dict)
        self.buffer['values'].append(value)
        self.buffer['terminated'].append(terminated)
        self.buffer['truncated'].append(truncated)
        
        self.current_episode_steps += 1
        
        # エピソード終了時に長さを記録
        if terminated or truncated:
            self.episode_lengths.append(self.current_episode_steps)
            self.current_episode_steps = 0

    def sample(self, batch_size):
        """
        ランダムにミニバッチをサンプリング
        
        Returns:
            batch: {
                'obs': np.array (batch_size, num_agents, obs_dim)
                'actions': np.array (batch_size, num_agents)
                'rewards': np.array (batch_size, num_agents)
                'global_states': np.array (batch_size, state_dim)
                'log_probs': np.array (batch_size, num_agents)
                'values': np.array (batch_size)
                'advantages': np.array (batch_size, num_agents)  # 事前に計算済み
            }
        """
        if len(self.buffer['obs']) == 0:
            return None
        
        # ランダムにインデックスを選択
        indices = np.random.choice(len(self.buffer['obs']), size=batch_size, replace=False)
        
        # 各項目をバッチ化
        obs_batch = []
        actions_batch = []
        rewards_batch = []
        log_probs_batch = []
        advantages_batch = []
        
        global_states_batch = []
        values_batch = []
        
        for idx in indices:
            # 観測・行動・報酬・log_probsをエージェント順に並べる
            obs_step = []
            actions_step = []
            rewards_step = []
            log_probs_step = []
            advantages_step = []
            
            for i in range(self.num_agents):
                agent_name = f'pursuer_{i}'
                obs_step.append(self.buffer['obs'][idx][agent_name])
                actions_step.append(self.buffer['actions'][idx][agent_name])
                rewards_step.append(self.buffer['rewards'][idx][agent_name])
                log_probs_step.append(self.buffer['log_probs'][idx][agent_name])
                
                # Advantageは事前に計算済みとする（後述のメソッドで追加）
                if 'advantages' in self.buffer and idx < len(self.buffer['advantages']):
                    advantages_step.append(self.buffer['advantages'][idx][agent_name])
                else:
                    advantages_step.append(0.0)  # 未計算なら0
            
            obs_batch.append(obs_step)
            actions_batch.append(actions_step)
            rewards_batch.append(rewards_step)
            log_probs_batch.append(log_probs_step)
            advantages_batch.append(advantages_step)
            
            global_states_batch.append(self.buffer['global_states'][idx])
            values_batch.append(self.buffer['values'][idx])
        
        batch = {
            'obs': np.array(obs_batch, dtype=np.float32),           # (batch, num_agents, obs_dim)
            'actions': np.array(actions_batch, dtype=np.int64),      # (batch, num_agents)
            'rewards': np.array(rewards_batch, dtype=np.float32),    # (batch, num_agents)
            'global_states': np.array(global_states_batch, dtype=np.float32),  # (batch, state_dim)
            'log_probs': np.array(log_probs_batch, dtype=np.float32),# (batch, num_agents)
            'values': np.array(values_batch, dtype=np.float32),      # (batch,)
            'advantages': np.array(advantages_batch, dtype=np.float32),  # (batch, num_agents)
        }
        
        return batch

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        """
        GAE（Generalized Advantage Estimation）でAdvantageを計算し、バッファに追加
        
        Args:
            gamma: 割引率
            gae_lambda: GAEのλパラメータ
        """
        advantages = []
        returns = []
        
        # エピソードごとに処理
        start_idx = 0
        for ep_len in self.episode_lengths:
            end_idx = start_idx + ep_len
            
            # エピソード内のデータを取得
            rewards_ep = [self.buffer['rewards'][i] for i in range(start_idx, end_idx)]
            values_ep = [self.buffer['values'][i] for i in range(start_idx, end_idx)]
            terminated_ep = [self.buffer['terminated'][i] for i in range(start_idx, end_idx)]
            truncated_ep = [self.buffer['truncated'][i] for i in range(start_idx, end_idx)]
            
            # 各エージェントごとにAdvantageを計算
            ep_advantages = []
            ep_returns = []
            
            for agent_idx in range(self.num_agents):
                agent_name = f'pursuer_{agent_idx}'
                
                # エージェントごとの報酬と価値を抽出
                agent_rewards = [r[agent_name] for r in rewards_ep]
                agent_values = values_ep.copy()  # グローバル状態から計算した価値は全エージェント共通
                
                # 終了時の次状態価値（0とする）
                next_value = 0.0
                
                # GAE計算
                delta = np.zeros(ep_len)
                for t in reversed(range(ep_len)):
                    if t == ep_len - 1:
                        # 最終ステップ
                        if terminated_ep[t] or truncated_ep[t]:
                            next_value = 0.0
                        else:
                            # 実際には次状態価値が必要だが、簡略化
                            next_value = agent_values[t]
                    
                    delta[t] = agent_rewards[t] + gamma * next_value - agent_values[t]
                    next_value = agent_values[t]
                
                # GAEの累積和
                advantage = np.zeros(ep_len)
                gae = 0.0
                for t in reversed(range(ep_len)):
                    gae = delta[t] + gamma * gae_lambda * gae
                    advantage[t] = gae
                
                ep_advantages.append(advantage)
                
                # Return（累積報酬）も計算しておく
                return_ep = np.zeros(ep_len)
                cumulative = 0.0
                for t in reversed(range(ep_len)):
                    cumulative = agent_rewards[t] + gamma * cumulative
                    return_ep[t] = cumulative
                ep_returns.append(return_ep)
            
            # エピソード内の各ステップにAdvantageとReturnを割り当て
            for t in range(ep_len):
                advantage_dict = {}
                return_dict = {}
                for agent_idx in range(self.num_agents):
                    agent_name = f'pursuer_{agent_idx}'
                    advantage_dict[agent_name] = ep_advantages[agent_idx][t]
                    return_dict[agent_name] = ep_returns[agent_idx][t]
                
                advantages.append(advantage_dict)
                returns.append(return_dict)
            
            start_idx = end_idx
        
        # バッファに追加
        self.buffer['advantages'] = advantages
        self.buffer['returns'] = returns

    def clear(self):
        """バッファをクリア"""
        for key in self.buffer:
            self.buffer[key] = []
        self.episode_lengths = []
        self.current_episode_steps = 0

    def __len__(self):
        return len(self.buffer['obs'])
    
# テスト実行
def test_pursuit_buffer():
    print("=== Pursuit + 環境ラッパ + メモリバッファ テスト開始 ===")
    
    # 環境とラッパの初期化
    env = PursuitWrapper(render_mode=None, max_cycles=50)  # テスト用に短く
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"エージェント数: {num_agents}")
    print(f"観測次元: {obs_dim}")
    print(f"グローバル状態次元: {state_dim}")
    print(f"行動空間サイズ: {action_dim}")
    
    # メモリバッファの初期化
    buffer = MultiAgentBuffer(num_agents, obs_dim, state_dim, action_dim)
    
    # 1エピソード実行（ランダムエージェント）
    env.reset()
    step_count = 0
    
    for agent in env.env.agent_iter():
        # 観測とグローバル状態を取得
        obs = env.get_obs(agent)
        global_state = env.get_global_state()
        
        if agent not in env.env.agents:
            action = None
            reward = 0.0
            terminated = True
            truncated = True
        else:
            _, _, terminated, truncated, _ = env.env.last(agent)
            if terminated or truncated:
                action = None
            else:
                # ランダム行動（テスト用）
                action = env.action_space.sample()
            
            # 1ステップ進める
            reward, terminated, truncated, info = env.step(agent, action)
        
        # 各エージェントの観測・行動・報酬・log_probを辞書でまとめる
        obs_dict = {}
        action_dict = {}
        reward_dict = {}
        log_prob_dict = {}
        
        for i in range(num_agents):
            agent_name = f'pursuer_{i}'
            if agent_name in env.env.agents:
                agent_obs = env.get_obs(agent_name)
                if agent_obs is not None:
                    obs_dict[agent_name] = agent_obs
                else:
                    obs_dict[agent_name] = np.zeros(obs_dim, dtype=np.float32)
                
                # 行動と報酬は現在のエージェントのみ実際の値、他は0またはNone
                if agent_name == agent:
                    action_dict[agent_name] = action if action is not None else 0
                    reward_dict[agent_name] = reward
                else:
                    action_dict[agent_name] = 0
                    reward_dict[agent_name] = 0.0
                
                # log_probはテスト用にランダム値（実際はポリシーから計算）
                log_prob_dict[agent_name] = np.log(1.0 / action_dim)  # 一様分布のlog_prob
            else:
                # deadエージェントは0埋め
                obs_dict[agent_name] = np.zeros(obs_dim, dtype=np.float32)
                action_dict[agent_name] = 0
                reward_dict[agent_name] = 0.0
                log_prob_dict[agent_name] = 0.0
        
        # Criticの価値推定（テスト用に0）
        value = 0.0
        
        # バッファに保存
        buffer.store(
            obs_dict, action_dict, reward_dict, global_state,
            log_prob_dict, value, terminated, truncated
        )
        
        step_count += 1
        if terminated or truncated:
            print(f"エピソード終了: {step_count}ステップ")
            break
    
    # バッファの状態を確認
    print(f"\nバッファに保存されたステップ数: {len(buffer)}")
    print(f"エピソード長さのリスト: {buffer.episode_lengths}")
    
    # Advantageを計算
    buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)
    print("Advantage計算完了")
    
    # ミニバッチをサンプリングして形状を確認
    batch_size = min(16, len(buffer))  # 小さいバッチでテスト
    batch = buffer.sample(batch_size)
    
    if batch is not None:
        print(f"\nミニバッチの形状:")
        print(f"obs: {batch['obs'].shape}")           # (batch, num_agents, obs_dim)
        print(f"actions: {batch['actions'].shape}")    # (batch, num_agents)
        print(f"rewards: {batch['rewards'].shape}")    # (batch, num_agents)
        print(f"global_states: {batch['global_states'].shape}")  # (batch, state_dim)
        print(f"log_probs: {batch['log_probs'].shape}") # (batch, num_agents)
        print(f"values: {batch['values'].shape}")      # (batch,)
        print(f"advantages: {batch['advantages'].shape}")  # (batch, num_agents)
        
        # 値の範囲を簡単に確認
        print(f"\n値の範囲（サンプル）:")
        print(f"rewards min/max: {batch['rewards'].min():.3f}, {batch['rewards'].max():.3f}")
        print(f"advantages min/max: {batch['advantages'].min():.3f}, {batch['advantages'].max():.3f}")
        print(f"log_probs min/max: {batch['log_probs'].min():.3f}, {batch['log_probs'].max():.3f}")
    else:
        print("バッファが空です")
    
    # バッファをクリア
    buffer.clear()
    print(f"\nバッファクリア後: {len(buffer)}")
    
    env.close()
    print("=== テスト終了 ===")