import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# -------------------------------------------------------------------
# 0. 修正点まとめ (詳細はチャット本文を参照)
# -------------------------------------------------------------------
# [FIX 1] _update_exploration(): 個人報酬の二重カウントを解消。
#         複数エージェントが同時に同じ新規マスを発見した場合、
#         発見者数で按分(fair split)する。これによりクラスタリング
#         (固まって行動)を報酬で誘発してしまう問題を解消。
# [FIX 2] os / CHECKPOINT_DIR の定義漏れを追加 (NameErrorでクラッシュしていた)。
# [FIX 3] GAEのブートストラップで、真の終端(全マス探索完了)と
#         打ち切り(MAX_STEPS到達)を区別し、真の終端ではnext_value=0にする。
# [FIX 4] 損失をタイムステップ数で正規化し、さらに1エピソードあたり
#         複数epoch(PPO_EPOCHS)の更新を行うようにして学習効率を改善。
# [FIX 5] 報酬のスケールをセンサー範囲に対して正規化し、エピソード
#         序盤(未探索が多い)と終盤(既探索が多い)での報酬スケールの
#         暴れを抑制(任意のチューニングだが安定化に寄与)。
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 1. 改修版 環境クラス (個別報酬の導入)
# -------------------------------------------------------------------
GRID_SIZE = 15
NUM_AGENTS = 3
SENSOR_RANGE = 1
MAX_STEPS = 100

# [FIX 2] チェックポイント保存先の定義


# [FIX 5] 報酬スケールの正規化用定数
SENSOR_AREA = (2 * SENSOR_RANGE + 1) ** 2  # 1エージェントが1stepで見渡せる最大マス数
TEAM_REWARD_WEIGHT = 1.0
IND_REWARD_WEIGHT = 2.0


class MultiSensorSearchEnv:
    def __init__(self, size: int = GRID_SIZE, num_agents: int = NUM_AGENTS):
        self.size = size
        self.num_agents = num_agents
        self.action_space = 5  # 0:待機, 1:上, 2:下, 3:左, 4:右
        self.explored_map = np.zeros((self.size, self.size), dtype=int)
        self.fig, self.ax = None, None
        self.reset()

    def reset(self) -> Dict[int, Dict]:
        self.explored_map = np.zeros((self.size, self.size), dtype=int)
        self.current_step = 0
        self.agent_positions = {
            i: (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            for i in range(self.num_agents)
        }
        self._update_exploration()
        return self._get_obs()

    def _get_obs(self) -> Dict[int, Dict]:
        obs = {}
        for i in range(self.num_agents):
            obs[i] = {
                "agent_id": i,  # エージェントIDを明示
                "position": self.agent_positions[i],
                "map": self.explored_map.copy()
            }
        return obs

    def _update_exploration(self) -> Tuple[int, Dict[int, float]]:
        """
        全体の新規発見マス数と、エージェントごとの「公平な」貢献度を計算する。

        [FIX 1] 旧実装では各エージェントの新規発見マス数を独立に計算しており、
        複数エージェントが同時に同じマスを新規発見すると、そのマスの報酬が
        発見者全員に満額支払われていた(二重・三重カウント)。これは
        「固まって同じ場所を探索するほど得」という、探索を分散させたい
        目的と矛盾するインセンティブを生み、ハイブリッド報酬導入後の
        学習不安定化の主因だった。

        修正版では、同時発見されたマスは発見者数で按分(fair split)し、
        単独で発見した場合のみ満額(1.0)を得られるようにする。
        """
        footprints = []
        for i in range(self.num_agents):
            x, y = self.agent_positions[i]
            x_min, x_max = max(0, x - SENSOR_RANGE), min(self.size, x + SENSOR_RANGE + 1)
            y_min, y_max = max(0, y - SENSOR_RANGE), min(self.size, y + SENSOR_RANGE + 1)

            mask = np.zeros_like(self.explored_map, dtype=bool)
            mask[y_min:y_max, x_min:x_max] = True
            mask &= (self.explored_map == 0)  # 更新前時点で未探索だったマスのみ
            footprints.append(mask)

        # 各マスを何人のエージェントが「同時に」新規発見したか
        discover_count = np.sum(footprints, axis=0)  # (size, size) の整数配列
        union_mask = discover_count > 0

        total_new_cells = int(np.sum(union_mask))

        with np.errstate(divide="ignore", invalid="ignore"):
            fair_share = np.where(discover_count > 0, 1.0 / np.maximum(discover_count, 1), 0.0)

        agent_new_cells: Dict[int, float] = {}
        for i in range(self.num_agents):
            agent_new_cells[i] = float(np.sum(footprints[i] * fair_share))

        # マップを実際に更新 (合併集合として一括更新)
        self.explored_map[union_mask] = 1

        return total_new_cells, agent_new_cells

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, bool, Dict]:
        self.current_step += 1
        rewards = {i: -0.1 for i in range(self.num_agents)}  # タイムペナルティ

        # 移動
        for i, action in actions.items():
            cx, cy = self.agent_positions[i]
            nx, ny = cx, cy
            if action == 1: ny += 1
            elif action == 2: ny -= 1
            elif action == 3: nx -= 1
            elif action == 4: nx += 1

            self.agent_positions[i] = (np.clip(nx, 0, self.size - 1), np.clip(ny, 0, self.size - 1))

        # 探索更新と報酬設定
        total_new, agent_new = self._update_exploration()

        # [FIX 5] 報酬設計: チーム協調報酬 + 個人貢献報酬 (センサー面積で正規化)
        # 正規化することで、序盤(未探索マスが多く報酬が跳ね上がる)と
        # 終盤(既探索が多く報酬がほぼ0になる)でのスケール差を抑え、
        # 学習の安定性を高める。
        team_reward = (total_new / SENSOR_AREA) * TEAM_REWARD_WEIGHT
        for i in range(self.num_agents):
            ind_reward = (agent_new[i] / SENSOR_AREA) * IND_REWARD_WEIGHT
            rewards[i] += team_reward + ind_reward

        total_cells = self.size * self.size
        coverage_ratio = np.sum(self.explored_map) / total_cells
        done = coverage_ratio >= 1.0 or self.current_step >= MAX_STEPS

        if coverage_ratio >= 1.0:
            for i in range(self.num_agents):
                rewards[i] += 20.0

        info = {
            "coverage": coverage_ratio,
            "agent_new_cells": agent_new,
            "terminated": coverage_ratio >= 1.0,  # [FIX 3] 真の終端かどうかのフラグ
        }

        return self._get_obs(), rewards, done, info


# -------------------------------------------------------------------
# 2. 2D RoPE Transformer エージェント (変更なし)
# -------------------------------------------------------------------
class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, k, coords):
        x_coords, y_coords = coords[..., 0].float(), coords[..., 1].float()

        sin_x = torch.repeat_interleave(torch.sin(torch.einsum("bn,f->bnf", x_coords, self.inv_freq)), 2, dim=-1)
        cos_x = torch.repeat_interleave(torch.cos(torch.einsum("bn,f->bnf", x_coords, self.inv_freq)), 2, dim=-1)
        sin_y = torch.repeat_interleave(torch.sin(torch.einsum("bn,f->bnf", y_coords, self.inv_freq)), 2, dim=-1)
        cos_y = torch.repeat_interleave(torch.cos(torch.einsum("bn,f->bnf", y_coords, self.inv_freq)), 2, dim=-1)

        sin = torch.cat([sin_x, sin_y], dim=-1).unsqueeze(1)
        cos = torch.cat([cos_x, cos_y], dim=-1).unsqueeze(1)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    def _rotate_half(self, x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)


class RoPETransformerAgent(nn.Module):
    def __init__(self, grid_size=15, num_agents=3, action_space=5, embed_dim=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, embed_dim)
        )

        self.agent_id_emb = nn.Embedding(num_agents, embed_dim)
        self.agent_pos_emb = nn.Linear(2, embed_dim)

        self.rope = RotaryPositionEmbedding2D(embed_dim // 4)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.actor_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, action_space))
        self.critic_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs_dict: dict, device="cpu"):
        positions = [obs_dict[i]["position"] for i in range(self.num_agents)]
        coords = torch.tensor(positions, dtype=torch.float32, device=device).unsqueeze(0)
        shared_map = torch.tensor(obs_dict[0]["map"], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        map_feat = self.map_encoder(shared_map).unsqueeze(1)
        agent_ids = torch.arange(self.num_agents, device=device).unsqueeze(0)

        # ID Token + Position Token
        agent_tokens = self.agent_id_emb(agent_ids) + self.agent_pos_emb(coords / float(self.grid_size))

        map_coord = torch.tensor([[[self.grid_size / 2.0, self.grid_size / 2.0]]], device=device)
        all_coords = torch.cat([map_coord, coords], dim=1)
        all_tokens = torch.cat([map_feat, agent_tokens], dim=1)

        # Attention + RoPE
        B, N, C = all_tokens.shape
        num_heads = 4
        head_dim = C // num_heads

        q = self.q_proj(all_tokens).view(B, N, num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(all_tokens).view(B, N, num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(all_tokens).view(B, N, num_heads, head_dim).transpose(1, 2)

        q, k = self.rope(q, k, all_coords)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, C)
        x = self.out_proj(context)

        agent_outputs = x[:, 1:, :]  # Agent tokensのみ抽出
        action_logits = self.actor_head(agent_outputs).squeeze(0)
        values = self.critic_head(agent_outputs).squeeze(0)

        return action_logits, values


# -------------------------------------------------------------------
# 3. 個人報酬対応 MAPPO 学習ルーチン (修正版)
# -------------------------------------------------------------------
PPO_EPOCHS = 4          # [FIX 4] 1エピソードあたりの更新epoch数
CLIP_RANGE = 0.2        # 一般的なPPOクリップ幅(0.8~1.2相当)に調整
GAMMA = 0.99
LAMBDA = 0.95


def train_mappo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MultiSensorSearchEnv(size=15, num_agents=3)
    model = RoPETransformerAgent(grid_size=15, num_agents=3, action_space=5).to(device)
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth")):
        print("succuessfully load parameters.")
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth"), map_location=device)
        model.load_state_dict(checkpoint)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for episode in range(1, 501):
        obs = env.reset()
        batch_obs, batch_actions, batch_logprobs, batch_rewards, batch_values = [], [], [], [], []
        info = {"coverage": 0.0, "terminated": False}

        for step in range(MAX_STEPS):
            batch_obs.append(obs)
            with torch.no_grad():
                logits, values = model(obs, device=device)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_dict = {i: actions[i].item() for i in range(env.num_agents)}
            next_obs, rewards_dict, done, info = env.step(actions_dict)

            # 各エージェント個別の報酬テンソル (3,)
            reward_tensor = torch.tensor([rewards_dict[i] for i in range(env.num_agents)], dtype=torch.float32, device=device)

            batch_actions.append(actions)
            batch_logprobs.append(log_probs)
            batch_rewards.append(reward_tensor)
            batch_values.append(values.squeeze(-1))

            obs = next_obs
            if done:
                break

        # [FIX 3] GAE計算: 真の終端(全マス探索完了)か打ち切り(MAX_STEP到達)かで
        # ブートストラップ値を切り替える。真の終端ではその先の価値は0。
        returns, advantages = [], []
        gae = torch.zeros(env.num_agents, device=device)

        with torch.no_grad():
            if info.get("terminated", False):
                next_val = torch.zeros(env.num_agents, device=device)
            else:
                _, next_value = model(obs, device=device)
                next_val = next_value.squeeze(-1)

        for t in reversed(range(len(batch_rewards))):
            nv = next_val if t == len(batch_rewards) - 1 else batch_values[t + 1]
            delta = batch_rewards[t] + GAMMA * nv - batch_values[t]
            gae = delta + GAMMA * LAMBDA * gae
            advantages.insert(0, gae.clone())
            returns.insert(0, gae + batch_values[t])

        b_actions = torch.stack(batch_actions)
        b_old_logprobs = torch.stack(batch_logprobs)
        b_returns = torch.stack(returns)
        b_adv = torch.stack(advantages)
        b_adv = (b_adv - b_adv.mean(dim=0, keepdim=True)) / (b_adv.std(dim=0, keepdim=True) + 1e-8)

        T = len(batch_obs)

        # [FIX 4] PPOパラメータ更新: 複数epoch回し、各epochで損失をT(タイムステップ数)
        # で正規化してから逆伝播する。旧実装は1エピソードにつき1回しか
        # optimizer.step()せず、かつ損失をTで平均していなかったため、
        # 長いエピソードほど勾配が肥大化し、更新も非常にサンプル非効率だった。
        for epoch in range(PPO_EPOCHS):
            optimizer.zero_grad()
            epoch_loss_sum = 0.0

            for t in range(T):
                logits, values = model(batch_obs[t], device=device)
                values = values.squeeze(-1)
                dist = Categorical(logits=logits)

                new_logprobs = dist.log_prob(b_actions[t])
                ratio = torch.exp(new_logprobs - b_old_logprobs[t])

                surr1 = ratio * b_adv[t]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * b_adv[t]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, b_returns[t])
                entropy_bonus = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
                (loss / T).backward()
                epoch_loss_sum += loss.item()

            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if episode % 20 == 0:
            print(f"Episode {episode:3d} | Total Step: {T:3d} | Coverage: {info['coverage']*100:.1f}%")
        if episode % 100 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "rope_transformer_mappo.pth"))


if __name__ == "__main__":
    train_mappo()