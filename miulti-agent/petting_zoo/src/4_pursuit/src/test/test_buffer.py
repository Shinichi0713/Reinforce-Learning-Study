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

# テスト実行
if __name__ == "__main__":
    test_pursuit_buffer()