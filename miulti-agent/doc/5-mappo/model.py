import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPPO_Trainer_Optimized:
    def __init__(self, obs_dim, action_dim, num_agents=2):
        self.num_agents = num_agents
        self.hidden_dim = 128
        
        # --- 工夫1: パラメータ共有 ---
        # Actorは1つだけ定義し、全エージェントで使い回す
        # 入力次元に Agent ID 分 (+num_agents) を追加
        self.actor = GRU_Actor(obs_dim + num_agents, action_dim, self.hidden_dim)
        
        # --- 工夫2: 集中Critic ---
        # 全員分の観測を合わせた次元を入力とする
        self.critic = GRU_Critic(obs_dim * num_agents, 256)
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def get_action(self, obs_list, h_actors):
        """
        obs_list: [agent0_obs, agent1_obs] (正規化済み)
        """
        actions = []
        log_probs = []
        new_h_actors = []
        
        for i in range(self.num_agents):
            # --- 工夫3: Agent ID の付与 ---
            agent_id = torch.zeros(self.num_agents)
            agent_id[i] = 1.0
            
            # 観測とIDを結合
            combined_obs = torch.cat([obs_list[i], agent_id], dim=-1).view(1, 1, -1)
            
            # 同一のActorネットワークで推論
            dist, h_a = self.actor(combined_obs, h_actors[i])
            action = dist.sample()
            
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
            new_h_actors.append(h_a)
            
        return actions, log_probs, new_h_actors

    def train(self, memory):
        # 集中Criticの学習: memory.states には [obs_agent1 + obs_agent2] が入っている前提
        # パラメータ共有されたActorの学習: 
        # 全エージェントの経験を一気に1つのバッチとしてActorに学習させる
        # (これにより学習データが実質2倍になり、収束が早まる)
        pass

# Actor更新時
torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
# Critic更新時
torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)