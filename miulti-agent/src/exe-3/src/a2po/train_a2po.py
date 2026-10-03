# -------------------------------------------------------------------
# 3. 個人報酬対応 MAPPO 学習ルーチン (修正版)
# -------------------------------------------------------------------
PPO_EPOCHS = 4          # [FIX 4] 1エピソードあたりの更新epoch数
CLIP_RANGE = 0.2        # 一般的なPPOクリップ幅(0.8~1.2相当)に調整
GAMMA = 0.99
LAMBDA = 0.95
NUM_TRAIN = 1500

"""
A2PO (Agent-by-agent Policy Optimization) の実装。

出典: Xihuai Wang et al., "Order Matters: Agent-by-agent Policy Optimization", ICLR 2023.
      (arXiv:2302.06205)

このファイルは multi_agent_search_mappo_fixed.py の環境・モデル定義を再利用し、
学習アルゴリズムだけをA2POに差し替えたものです。

--------------------------------------------------------------------------
[重要] 元論文の理論設定とこの実装の違いについて
--------------------------------------------------------------------------
A2POの monotonic improvement (単調改善) の理論は、「単一の共有報酬 r(s,a) /
単一の価値関数 V(s)」を持つ DEC-MDP を前提に構築されています(論文 Sec.3.1)。

一方、本プロジェクトの環境はエージェントごとに異なる報酬(個人貢献報酬 + チーム
協調報酬のハイブリッド)を持ち、モデルも各エージェントが個別の価値関数
(critic_head の出力はエージェントごとに独立)を持つ設計になっています。

そのため本実装では、論文のPre-OPC (preceding-agent off-policy correction) を
「各エージェント自身の報酬・価値関数」に対して適用する実務的な拡張として実装
しています。これはHAPPO/A2POを個別報酬・分散型critic設定に拡張する際の一般的な
やり方(IPPO/MAPPOが単一報酬の理論を個別報酬に拡張しているのと同様のスタンス)
ですが、論文が証明する厳密な単調改善保証はそのままでは成立しない点にご注意
ください。それでも、
  - preceding agentの方策変化をRetrace風に補正してから advantage を計算する
  - advantageの絶対値が大きいエージェントから優先的に(確率的に)更新する
  - 更新順位に応じてクリップ幅を適応的に広げる
という3つの設計思想はそのまま活かされており、同時更新(MAPPO)で起きがちな
非定常性問題を緩和する効果が期待できます。
--------------------------------------------------------------------------

モデルはパラメータ共有版(論文 Algorithm 2)として実装しています。本プロジェクトの
RoPETransformerAgent はもともと1回のforwardで全エージェント分のlogits/valueを
出力する「共有ネットワーク」構造なので、Algorithm 2 (parameter sharing) と
自然に対応します。
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_agent_search_mappo_fixed import (
    MultiSensorSearchEnv,
    RoPETransformerAgent,
    GRID_SIZE,
    NUM_AGENTS,
    MAX_STEPS,
    NUM_LAYERS,
    NUM_HEADS,
    FF_MULT,
    CHECKPOINT_DIR,
    GAMMA,
    LAMBDA,
)

# -------------------------------------------------------------------
# A2PO ハイパーパラメータ
# -------------------------------------------------------------------
A2PO_STAGES = 2            # 1イテレーションあたりの「全エージェントを1巡させる」回数
                            # (論文の "for n epochs do" に相当。大きいほど更新回数が
                            #  増え学習は進みやすいが、計算コストも比例して増える)
BASE_CLIP_EPS = 0.2         # 基本クリップ幅 ε
ADAPTIVE_CLIP_C = 0.5       # クリップ幅の適応係数 c_ε (論文は全タスクで0.5を使用)
SEMI_GREEDY_PROB = 0.5      # semi-greedy選択でランダムに選ぶ確率 (残り0.5は|advantage|最大を選択)


def adaptive_clip_eps(base_eps: float, order_k: int, num_agents: int, c_eps: float = ADAPTIVE_CLIP_C) -> float:
    """
    論文 Sec.4 の adaptive clipping parameter: C(eps, k) = eps*c + eps*(1-c)*(k/n)
    order_k は 1-indexed の更新順位 (1番目に更新されるエージェントは k=1)。
    """
    return base_eps * c_eps + base_eps * (1.0 - c_eps) * (order_k / num_agents)


def train_a2po():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MultiSensorSearchEnv(size=GRID_SIZE, num_agents=NUM_AGENTS)
    model = RoPETransformerAgent(
        grid_size=GRID_SIZE, num_agents=NUM_AGENTS, action_space=env.action_space,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_mult=FF_MULT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for episode in range(1, 501):
        # --------------------------------------------------------
        # 1. ロールアウト収集 (このイテレーションの「基準方策 π」を固定)
        # --------------------------------------------------------
        obs = env.reset()
        batch_obs, batch_actions, batch_old_logprobs, batch_rewards, batch_old_values = [], [], [], [], []
        info = {"coverage": 0.0, "terminated": False}

        for step in range(MAX_STEPS):
            batch_obs.append(obs)
            with torch.no_grad():
                logits, values = model(obs, device=device)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                old_logprobs = dist.log_prob(actions)

            actions_dict = {i: actions[i].item() for i in range(env.num_agents)}
            next_obs, rewards_dict, done, info = env.step(actions_dict)
            reward_tensor = torch.tensor(
                [rewards_dict[i] for i in range(env.num_agents)], dtype=torch.float32, device=device
            )

            batch_actions.append(actions)
            batch_old_logprobs.append(old_logprobs)
            batch_rewards.append(reward_tensor)
            batch_old_values.append(values.squeeze(-1))

            obs = next_obs
            if done:
                break

        T = len(batch_obs)
        b_actions = torch.stack(batch_actions)          # (T, num_agents)
        b_old_logprobs = torch.stack(batch_old_logprobs)  # (T, num_agents)  : log π(a|s)  (基準方策 π)
        b_rewards = torch.stack(batch_rewards)            # (T, num_agents)
        b_old_values = torch.stack(batch_old_values)       # (T, num_agents) : V(s) 基準方策下での価値

        with torch.no_grad():
            if info.get("terminated", False):
                next_val = torch.zeros(env.num_agents, device=device)
            else:
                _, next_value = model(obs, device=device)
                next_val = next_value.squeeze(-1)

        # --------------------------------------------------------
        # 2. A2PO 本体: ステージを A2PO_STAGES 回繰り返し、
        #    各ステージ内で全エージェントを1回ずつ agent-by-agent に更新する
        # --------------------------------------------------------
        for stage in range(A2PO_STAGES):
            remaining = set(range(env.num_agents))  # このステージでまだ更新していないエージェント
            preceding = []                            # このステージで既に更新したエージェント(更新順に格納)

            for order_k in range(1, env.num_agents + 1):
                # ---- (a) 現在(ステージ途中)の方策での全エージェント分 log-prob / value を評価 ----
                # preceding-agent off-policy correction と agent selection の両方に必要。
                # 勾配計算は不要なので no_grad で軽量に評価する。
                with torch.no_grad():
                    cur_logprobs = torch.zeros(T, env.num_agents, device=device)
                    cur_values = torch.zeros(T, env.num_agents, device=device)
                    for t in range(T):
                        logits, values = model(batch_obs[t], device=device)
                        dist = Categorical(logits=logits)
                        cur_logprobs[t] = dist.log_prob(b_actions[t])
                        cur_values[t] = values.squeeze(-1)

                    # preceding agents の joint policy ratio (打ち切り: min(1, ratio))
                    if len(preceding) == 0:
                        c = torch.ones(T, device=device)
                    else:
                        ratio_preceding = torch.exp(
                            cur_logprobs[:, preceding].sum(dim=1) - b_old_logprobs[:, preceding].sum(dim=1)
                        )
                        c = torch.clamp(ratio_preceding, max=1.0)

                    # ---- (b) Preceding-agent Off-Policy Correction (Pre-OPC) によるアドバンテージ推定 ----
                    # Retrace(λ)型の再帰: A(t) = delta(t) + gamma*lambda*c(t+1)*A(t+1)
                    # c(t+1) は preceding agents の方策変化を打ち切り重要度サンプリングで補正する項。
                    # preceding が空集合のときは c=1 となり、通常のGAEと一致する。
                    adv = torch.zeros(T, env.num_agents, device=device)
                    gae = torch.zeros(env.num_agents, device=device)
                    for t in reversed(range(T)):
                        nv = next_val if t == T - 1 else cur_values[t + 1]
                        delta = b_rewards[t] + GAMMA * nv - cur_values[t]
                        c_next = c[t + 1] if t + 1 < T else torch.tensor(1.0, device=device)
                        gae = delta + GAMMA * LAMBDA * c_next * gae
                        adv[t] = gae

                # ---- (c) Semi-greedy agent selection rule ----
                scores = {i: adv[:, i].abs().mean().item() for i in remaining}
                if random.random() < SEMI_GREEDY_PROB:
                    i_sel = random.choice(list(remaining))
                else:
                    i_sel = max(remaining, key=lambda a: scores[a])
                remaining.discard(i_sel)

                # ---- (d) Adaptive clipping parameter ----
                eps_i = adaptive_clip_eps(BASE_CLIP_EPS, order_k, env.num_agents)
                g_half = eps_i / 2.0  # preceding joint ratio には半分の幅でクリップ(論文Sec.4)

                # ---- (e) 選ばれたエージェント i_sel の方策・価値関数を更新 ----
                optimizer.zero_grad()
                for t in range(T):
                    logits, values = model(batch_obs[t], device=device)
                    dist = Categorical(logits=logits)
                    logp_all = dist.log_prob(b_actions[t])  # (num_agents,) 勾配あり

                    new_logprob_i = logp_all[i_sel]
                    ratio_i = torch.exp(new_logprob_i - b_old_logprobs[t, i_sel])

                    # preceding joint ratio (このagent iの勾配には影響させない: 定数として扱う)
                    with torch.no_grad():
                        if len(preceding) == 0:
                            g_val = torch.tensor(1.0, device=device)
                        else:
                            logp_preceding = logp_all[preceding].detach()
                            ratio_preceding_t = torch.exp(logp_preceding - b_old_logprobs[t, preceding])
                            joint_ratio_preceding = ratio_preceding_t.prod()
                            g_val = torch.clamp(joint_ratio_preceding, 1.0 - g_half, 1.0 + g_half)

                    l_ratio = ratio_i * g_val
                    A_i = adv[t, i_sel]  # Pre-OPCで補正済み・勾配なし

                    surr1 = l_ratio * A_i
                    surr2 = torch.clamp(l_ratio, 1.0 - eps_i, 1.0 + eps_i) * A_i
                    actor_loss = -torch.min(surr1, surr2)

                    value_target = (A_i + cur_values[t, i_sel]).detach()
                    critic_loss = F.mse_loss(values.squeeze(-1)[i_sel], value_target)

                    entropy_bonus = dist.entropy()[i_sel]

                    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
                    (loss / T).backward()

                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                preceding.append(i_sel)

        if episode % 20 == 0:
            print(f"Episode {episode:3d} | Total Step: {T:3d} | Coverage: {info['coverage']*100:.1f}%")
        if episode % 100 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "rope_transformer_a2po.pth"))


if __name__ == "__main__":
    train_a2po()
