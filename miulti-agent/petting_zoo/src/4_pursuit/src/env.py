import numpy as np
from pettingzoo.sisl import pursuit_v4

class PursuitWrapper:
    def __init__(self, render_mode=None, max_cycles=500, obs_range=7):
        self.env = pursuit_v4.env(
            render_mode=render_mode,
            max_cycles=max_cycles,
            shared_reward=True
        )
        self.obs_range = obs_range
        self.possible_agents = self.env.possible_agents
        self.num_agents = len(self.possible_agents)
        
        self.obs_dim = (obs_range * obs_range * 3)
        self.state_dim = self.obs_dim * self.num_agents
        
        self.action_space = self.env.action_space(self.possible_agents[0])
        self.action_dim = self.action_space.n

    def reset(self):
        """環境をリセット（MAPPO側で agent_iter を回す前提）"""
        self.env.reset()

    def get_obs(self, agent):
        """指定したエージェントの観測を取得し、フラット化する"""
        # agent がまだ存在するか確認
        if agent not in self.env.agents:
            return None
        
        obs, _, _, _, _ = self.env.last(agent)
        if obs is None:
            return None
        obs_flat = obs.reshape(-1).astype(np.float32)
        return obs_flat

    def get_global_state(self):
        """全エージェントの観測を結合してグローバル状態を作る"""
        obs_list = []
        for agent in self.possible_agents:
            # 存在するエージェントのみ観測を取得
            if agent in self.env.agents:
                obs = self.get_obs(agent)
                if obs is not None:
                    obs_list.append(obs)
        if not obs_list:
            return None
        global_state = np.concatenate(obs_list).astype(np.float32)
        return global_state

    def step(self, agent, action):
        """
        指定したエージェントに対して1ステップ進める
        
        Args:
            agent: 対象エージェント名
            action: 行動（dead の場合は None）
        
        Returns:
            reward, terminated, truncated, info
        """
        # agent がまだ存在するか確認
        if agent not in self.env.agents:
            # すでに dead なら何もしない
            return 0.0, True, True, {}
        
        # dead チェック
        _, _, terminated, truncated, _ = self.env.last(agent)
        if terminated or truncated:
            step_action = None
        else:
            step_action = action
        
        self.env.step(step_action)
        
        # 再度存在確認（step で削除される可能性あり）
        if agent not in self.env.agents:
            return 0.0, True, True, {}
        
        _, reward, terminated, truncated, info = self.env.last(agent)
        return reward, terminated, truncated, info

    def close(self):
        self.env.close()