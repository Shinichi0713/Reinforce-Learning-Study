from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


# ==========================================
# 追加: エージェント順序のユーティリティ
# ==========================================
def permute_along_agent_dim(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """
    x: (B, N, ...)  order: (B, N)  の順序でエージェント次元を並べ替える
    order[b, i] = 「i番目のデコードスロットに入る、元のエージェントindex」
    """
    B, N = order.shape
    idx = order.view(B, N, *([1] * (x.dim() - 2))).expand_as(x) if x.dim() > 2 else order
    return torch.gather(x, dim=1, index=idx)


def unpermute_along_agent_dim(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """
    permute_along_agent_dim の逆変換。
    デコード順に並んだ x を、元のエージェントID順に戻す。
    """
    B, N = order.shape
    inv_order = torch.argsort(order, dim=1)  # 逆置換
    idx = inv_order.view(B, N, *([1] * (x.dim() - 2))).expand_as(x) if x.dim() > 2 else inv_order
    return torch.gather(x, dim=1, index=idx)


def random_agent_order(B: int, N: int, device) -> torch.Tensor:
    """バッチ内サンプルごとに独立なランダム順序を生成"""
    order = torch.stack([torch.randperm(N, device=device) for _ in range(B)], dim=0)
    return order


def priority_agent_order(priority_scores: torch.Tensor) -> torch.Tensor:
    """
    priority_scores: (B, N) 値が小さいほど優先度が高い（例: 最寄りの未捕獲ターゲットまでの距離）
    捕獲済みなど「優先度なし」のエージェントは呼び出し側で十分大きな値（例: 1e6）にしておく。
    戻り値 order: (B, N) スコア昇順（優先度の高い順）のエージェントindex列
    """
    order = torch.argsort(priority_scores, dim=1)  # 小さい順 = 優先度が高い順
    return order

# ==========================================
# 1. 🌟 新設: Gated Multi-Head Attention (Attention-Sink-Free)
# ==========================================
class GatedCrossAttention(nn.Module):
    """
    Cross-Attention用ゲート付きAttention。
    ゲートは query と memory(key=value)の要約の両方から計算する。
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)  # 🌟 query + memory要約
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        # query: (B, T, d_model), key == value == memory: (B, S, d_model)
        B, T, _ = query.shape
        S = key.shape[1]

        q = self.q_proj(query).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # 🌟 memory(key引数=projection前のmemoryテンソル)を平均して要約
        memory_summary = key.mean(dim=1, keepdim=True).expand(-1, T, -1)  # (B, T, d_model)
        gate_input = torch.cat([query, memory_summary], dim=-1)           # (B, T, d_model*2)
        gate_scores = torch.sigmoid(self.gate_proj(gate_input))           # (B, T, d_model)
        gate_scores = gate_scores.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
            else:
                attn_logits = attn_logits + attn_mask

        attn_probs = F.softmax(attn_logits, dim=-1)
        sdpa_out = torch.matmul(attn_probs, v)

        gated_out = sdpa_out * gate_scores
        gated_out = gated_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(gated_out), attn_probs

class GatedMultiheadAttention(nn.Module):
    """
    "Gated Attention for Large Language Models" に基づく
    SDPAの直後にヘッド固有のSigmoidゲートを適用するアテンション機構
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        # Q, K, V の射影層
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 🌟 論文の肝: Query（または入力）から各ヘッド固有のゲートスコア（スカラーまたはベクトル）を計算する層
        # 最も安定して効果の高い、ヘッドごとの次元に合わせる射影を設定
        self.gate_proj = nn.Linear(d_model, d_model)

        # 出力の射影層
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        # query, key, value: (B, T, d_model)
        B, T, _ = query.shape
        S = key.shape[1] # Keyのシーケンス長

        # 1. 線形投影およびヘッド分割 (B, nhead, T, head_dim)
        q = self.q_proj(query).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # 2. ゲートスコアの計算 (Query依存のシグモイド)
        # 論文の数式: O = SDPA(Q,K,V) ⊙ σ(XW) に準拠
        gate_scores = torch.sigmoid(self.gate_proj(query)) # (B, T, d_model)
        gate_scores = gate_scores.view(B, T, self.nhead, self.head_dim).transpose(1, 2) # (B, nhead, T, head_dim)

        # 3. Scaled Dot-Product Attention (SDPA)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if attn_mask is not None:
            # 決定論的マスク/因果関係マスクの適用
            if attn_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(attn_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
            else:
                attn_logits = attn_logits + attn_mask

        attn_probs = F.softmax(attn_logits, dim=-1)
        sdpa_out = torch.matmul(attn_probs, v) # (B, nhead, T, head_dim)

        # 4. 🌟 ゲートの適用 (ここで Attention Sink が除去され、Sparsityが生まれる)
        gated_out = sdpa_out * gate_scores

        # 5. ヘッドの結合と最終投影
        gated_out = gated_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(gated_out), attn_probs


# ==========================================
# 2. MoE (Top-1 Gating) モジュール
# ==========================================
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=4, expert_hidden_dim=128):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_hidden_dim),
                nn.GELU(),
                nn.Linear(expert_hidden_dim, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)

        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top1_probs, top1_indices = torch.topk(gate_probs, k=1, dim=-1)

        # --- 追加: 負荷分散損失（Switch Transformer方式）---
        num_tokens = x_flat.shape[0]
        one_hot = F.one_hot(top1_indices.squeeze(-1), num_classes=self.num_experts).float()
        frac_tokens_per_expert = one_hot.mean(dim=0)          # 各expertに実際に割り当てられた割合
        frac_prob_per_expert = gate_probs.mean(dim=0)          # 各expertのゲート確率の平均
        aux_loss = self.num_experts * (frac_tokens_per_expert * frac_prob_per_expert).sum()
        self.last_aux_loss = aux_loss  # 後でupdate側で回収

        output_flat = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            mask = (top1_indices.squeeze(-1) == expert_idx)
            if not mask.any():
                continue
            expert_out = self.experts[expert_idx](x_flat[mask])
            output_flat[mask] = expert_out * gate_probs[mask, expert_idx:expert_idx+1]

        return output_flat.view(orig_shape)


# ==========================================
# 3. 🌟 Gated Attention 統合型 Transformer レイヤー
# ==========================================
class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_experts=4, dim_feedforward=128):
        super().__init__()
        # nn.MultiheadAttention から GatedMultiheadAttention へ変更
        self.self_attn = GatedMultiheadAttention(d_model, nhead)
        self.moe = MoELayer(d_model, num_experts=num_experts, expert_hidden_dim=dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask=None):
        x = self.norm1(src)
        # 自作アテンションのため引数を明示指定
        attn_out, _ = self.self_attn(query=x, key=x, value=x, attn_mask=src_mask)
        src = src + self.dropout(attn_out)

        x = self.norm2(src)
        moe_out = self.moe(x)
        src = src + self.dropout(moe_out)
        return src


class MoETransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_experts=4, dim_feedforward=128):
        super().__init__()
        self.self_attn = GatedMultiheadAttention(d_model, nhead)       # Self-Attentionは従来通り
        self.multihead_attn = GatedCrossAttention(d_model, nhead)      # 🌟 Cross-Attentionはこちらに変更
        self.moe = MoELayer(d_model, num_experts=num_experts, expert_hidden_dim=dim_feedforward)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, tgt_mask=None):
        x = self.norm1(tgt)
        attn_out, _ = self.self_attn(query=x, key=x, value=x, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(attn_out)

        x = self.norm2(tgt)
        attn_out2, _ = self.multihead_attn(query=x, key=memory, value=memory, attn_mask=None)
        tgt = tgt + self.dropout(attn_out2)

        x = self.norm3(tgt)
        moe_out = self.moe(x)
        tgt = tgt + self.dropout(moe_out)
        return tgt


# ==========================================
# 3. MoE 対応 MAT 各コンポーネント
# ==========================================
class MATObsEncoder(nn.Module):
    def __init__(self, obs_range=7, in_channels=4, d_model=64, nhead=4,
                 num_layers=2, num_agents=8, num_experts=4):
        super().__init__()
        self.obs_range = obs_range
        self.num_tokens = obs_range * obs_range
        self.spatial_dim = obs_range * obs_range * in_channels
        self.num_agents = num_agents

        self.embedding = nn.Linear(in_channels, d_model)

        pos_emb = self._get_2d_sin_cos_embedding(obs_range, d_model)
        self.register_buffer("pos_embedding", pos_emb)

        self.action_history_embed = nn.Sequential(
            nn.Linear(num_agents * 5, d_model),
            nn.GELU(),
        )
        self.feature_fuse = nn.Linear(d_model * 2, d_model)

        # 🌟 修正: 標準の Transformer を MoE 対応のレイヤーで構築
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(d_model=d_model, nhead=nhead, num_experts=num_experts, dim_feedforward=d_model * 2)
            for _ in range(num_layers)
        ])

    def _get_2d_sin_cos_embedding(self, grid_size, d_model):
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sin-cos embedding"
        y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
        y = y.flatten().float()
        x = x.flatten().float()

        d_feat = d_model // 2
        omega = torch.exp(torch.arange(0, d_feat, 2).float() * -(np.log(10000.0) / d_feat))

        out_y_sin = torch.sin(torch.outer(y, omega))
        out_y_cos = torch.cos(torch.outer(y, omega))
        out_x_sin = torch.sin(torch.outer(x, omega))
        out_x_cos = torch.cos(torch.outer(x, omega))

        pos_emb = torch.cat([out_y_sin, out_y_cos, out_x_sin, out_x_cos], dim=-1)
        return pos_emb.unsqueeze(0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        spatial_obs = obs[:, :self.spatial_dim]
        action_history = obs[:, self.spatial_dim:]

        x = spatial_obs.view(B, self.num_tokens, -1)
        x = self.embedding(x) + self.pos_embedding

        # MoE 層を順に伝播
        for layer in self.layers:
            x = layer(x)

        spatial_feature = x.mean(dim=1)
        act_emb = self.action_history_embed(action_history)
        fused = torch.cat([spatial_feature, act_emb], dim=-1)
        return self.feature_fuse(fused)


class MATEncoder(nn.Module):
    def __init__(self, num_agents=8, d_model=64, nhead=4, num_layers=2, num_experts=4):
        super().__init__()
        self.num_agents = num_agents
        self.agent_pos_embedding = nn.Parameter(torch.randn(1, num_agents, d_model))

        # 🌟 修正: エージェント間トランスフォーマーを MoE 化
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(d_model=d_model, nhead=nhead, num_experts=num_experts, dim_feedforward=d_model * 2)
            for _ in range(num_layers)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, agent_feats: torch.Tensor):
        x = agent_feats + self.agent_pos_embedding

        for layer in self.layers:
            x = layer(x)

        values = self.value_head(x).squeeze(-1)
        return x, values


class MATDecoder(nn.Module):
    def __init__(self, num_agents=8, action_dim=5, d_model=64, nhead=4, num_layers=2, num_experts=4):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.START = action_dim

        self.action_embedding = nn.Embedding(action_dim + 1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_agents, d_model))

        # 🌟 修正: デコーダ側トランスフォーマーも MoE 化
        self.layers = nn.ModuleList([
            MoETransformerDecoderLayer(d_model=d_model, nhead=nhead, num_experts=num_experts, dim_feedforward=d_model * 2)
            for _ in range(num_layers)
        ])

        self.action_head = nn.Linear(d_model, action_dim)

        mask = torch.triu(torch.ones(num_agents, num_agents, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, enc_out: torch.Tensor, shifted_actions: torch.Tensor) -> torch.Tensor:
        tgt = self.action_embedding(shifted_actions) + self.pos_embedding[:, :shifted_actions.shape[1], :]
        mask = self.causal_mask[:shifted_actions.shape[1], :shifted_actions.shape[1]]

        # デコーダのループ処理
        for layer in self.layers:
            tgt = layer(tgt, memory=enc_out, tgt_mask=mask)

        logits = self.action_head(tgt)
        return logits

    @torch.no_grad()
    def autoregressive_decode(self, enc_out: torch.Tensor, greedy: bool = False):
        B = enc_out.shape[0]
        device = enc_out.device

        shifted_actions = torch.full((B, self.num_agents), self.START, dtype=torch.long, device=device)
        actions = torch.zeros((B, self.num_agents), dtype=torch.long, device=device)
        log_probs = torch.zeros((B, self.num_agents), device=device)

        for i in range(self.num_agents):
            tgt = self.action_embedding(shifted_actions) + self.pos_embedding

            mask = self.causal_mask

            # デコードステップも MoE レイヤーを順に通過
            cur_tgt = tgt
            for layer in self.layers:
                cur_tgt = layer(cur_tgt, memory=enc_out, tgt_mask=mask)

            logits_i = self.action_head(cur_tgt[:, i, :])
            dist_i = Categorical(logits=logits_i)

            a_i = torch.argmax(logits_i, dim=-1) if greedy else dist_i.sample()

            actions[:, i] = a_i
            log_probs[:, i] = dist_i.log_prob(a_i)

            if i + 1 < self.num_agents:
                shifted_actions[:, i + 1] = a_i

        return actions, log_probs


class MATActorCritic(nn.Module):
    def __init__(self, obs_range=7, in_channels=4, d_model=64, nhead=4,
                 spatial_layers=2, enc_layers=2, dec_layers=2,
                 act_dim=5, num_agents=8):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = act_dim
        self.obs_dim = obs_range * obs_range * in_channels + num_agents * 5

        self.obs_encoder = MATObsEncoder(obs_range, in_channels, d_model, nhead,
                                         spatial_layers, num_agents)
        self.encoder = MATEncoder(num_agents, d_model, nhead, enc_layers)
        self.decoder = MATDecoder(num_agents, act_dim, d_model, nhead, dec_layers)

    def encode(self, joint_obs: torch.Tensor, order: torch.Tensor = None):
        """
        order: (B, N) 指定があれば、その順序でエージェント次元を並べ替えてからエンコードする。
        Noneなら並べ替えなし（従来通りのID順）。
        """
        B, N, D = joint_obs.shape

        if order is not None:
            joint_obs = permute_along_agent_dim(joint_obs, order)

        flat_obs = joint_obs.reshape(B * N, D)
        agent_feats = self.obs_encoder(flat_obs).view(B, N, -1)
        enc_out, values = self.encoder(agent_feats)
        # valuesも並べ替えられた順で出てくるので、後で使う側は必要に応じてunpermuteする
        return enc_out, values

    def forward_train(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor,
                       order: torch.Tensor = None):
        """
        order: (B, N) 学習時に使った（rolloutで実際に使われた）順序。
        rollout時にorderを記録しておき、学習時は必ず同じorderを再現する必要がある
        （でないとcausal maskの意味とactionの対応がズレる）。
        """
        enc_out, values = self.encode(joint_obs, order=order)

        B = joint_obs.shape[0]

        # joint_actionsもorderに合わせて並べ替える
        if order is not None:
            actions_for_decode = permute_along_agent_dim(joint_actions, order)
        else:
            actions_for_decode = joint_actions

        start_col = torch.full((B, 1), self.decoder.START, dtype=torch.long, device=joint_obs.device)
        shifted_actions = torch.cat([start_col, actions_for_decode[:, :-1]], dim=1)

        logits = self.decoder(enc_out, shifted_actions)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions_for_decode)
        entropy = dist.entropy()

        # 呼び出し側（元のエージェントID順）に揃えて返す
        if order is not None:
            log_probs = unpermute_along_agent_dim(log_probs, order)
            entropy = unpermute_along_agent_dim(entropy, order)
            values = unpermute_along_agent_dim(values, order)

        return log_probs, entropy, values

    @torch.no_grad()
    def act(self, joint_obs: torch.Tensor, order: torch.Tensor = None, greedy: bool = False):
        enc_out, values = self.encode(joint_obs, order=order)
        actions, log_probs = self.decoder.autoregressive_decode(enc_out, greedy=greedy)

        # デコード順（order順）で出てくるので、元のエージェントID順に戻す
        if order is not None:
            actions = unpermute_along_agent_dim(actions, order)
            log_probs = unpermute_along_agent_dim(log_probs, order)
            values = unpermute_along_agent_dim(values, order)

        return actions, log_probs, values


class MAT_PPO:
    def __init__(self, num_agents=8, obs_dim=236, action_dim=5,
                 d_model=64, nhead=4, spatial_layers=2, enc_layers=2, dec_layers=2,
                 lr=1e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 order_mode="random",  # 🌟 追加: "fixed" | "random" | "priority"
                 device=torch.device("cpu")):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.order_mode = order_mode
        self.device = device

        self.model = MATActorCritic(
            d_model=d_model, nhead=nhead,
            spatial_layers=spatial_layers, enc_layers=enc_layers, dec_layers=dec_layers,
            act_dim=action_dim, num_agents=num_agents,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _make_order(self, B: int, priority_scores: np.ndarray = None) -> torch.Tensor:
        """
        order_modeに応じて (B, N) の並べ替えインデックスを生成する。
        priority_scores: (B, N) or (N,) の優先度スコア（値が小さいほど優先）。
                          order_mode="priority" のときに必須。
        """
        if self.order_mode == "fixed":
            return torch.arange(self.num_agents, device=self.device).unsqueeze(0).expand(B, -1)

        elif self.order_mode == "random":
            return random_agent_order(B, self.num_agents, self.device)

        elif self.order_mode == "priority":
            assert priority_scores is not None, "priority_scores が必要です（order_mode='priority'）"
            scores_t = torch.as_tensor(priority_scores, dtype=torch.float32, device=self.device)
            if scores_t.dim() == 1:
                scores_t = scores_t.unsqueeze(0).expand(B, -1)
            return priority_agent_order(scores_t)

        else:
            raise ValueError(f"unknown order_mode: {self.order_mode}")

    def get_action(self, joint_obs: np.ndarray, priority_scores: np.ndarray = None, greedy: bool = False):
        """
        priority_scores: 環境側で計算した「各エージェントの最寄り未捕獲ターゲットまでの距離」など。
                          (num_agents,) の配列。order_mode="priority" のとき使用。
                          未指定なら order_mode に従う（推論時は基本 "priority" か "fixed" 推奨。
                          "random" は学習時の探索目的で使うのが基本）。
        """
        obs_t = torch.as_tensor(joint_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        order = self._make_order(B=1, priority_scores=priority_scores)

        actions, log_probs, values = self.model.act(obs_t, order=order, greedy=greedy)
        return (
            actions.squeeze(0).cpu().numpy(),
            log_probs.squeeze(0).cpu().numpy(),
            values.squeeze(0).cpu().numpy(),
            order.squeeze(0).cpu().numpy(),  # 🌟 rolloutバッファに保存しておく（学習時に再利用するため）
        )

    def update(self, batch: dict, epochs: int = 3, use_individual_clipping: bool = True):
        """
        use_individual_clipping:
            True  -> エージェントごとに比率・アドバンテージを個別クリッピング(推奨・新方式)
            False -> 従来通りチーム全体の同時比率でクリッピング(比較用)
        """
        if batch is None:
            return 0.0, 0.0, 0.0

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)  # (B, N)
        returns = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        order = torch.as_tensor(batch["order"], dtype=torch.long, device=self.device)

        # 🌟 変更点: アドバンテージの正規化を、エージェント次元も含めた全体で行う
        #    (個別方式でもチーム方式でも、正規化のスケール基準は揃えておく)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(epochs):
            new_log_probs, entropy, values = self.model.forward_train(obs, actions, order=order)
            # new_log_probs, entropy: (B, N)  values: (B, N)

            if use_individual_clipping:
                # 🌟 新方式: エージェントごとに比率とアドバンテージを個別評価
                per_agent_ratio = torch.exp(new_log_probs - old_log_probs)  # (B, N)

                surr1 = per_agent_ratio * advantages
                surr2 = torch.clamp(per_agent_ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

                # 各サンプル・各エージェントでmin(surr1, surr2)を取り、全体平均
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            else:
                # 従来方式: チーム全体の同時確率比率でクリッピング(比較用に残す)
                joint_new_log_prob = new_log_probs.sum(dim=-1)
                joint_old_log_prob = old_log_probs.sum(dim=-1)
                ratio = torch.exp(joint_new_log_prob - joint_old_log_prob)

                step_advantages = advantages.mean(dim=-1)

                surr1 = ratio * step_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * step_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            if values.shape != returns.shape:
                returns_reshaped = returns.view_as(values)
            else:
                returns_reshaped = returns

            critic_loss = nn.SmoothL1Loss()(values, returns_reshaped) * self.value_coef

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy.mean().item())

        return float(np.mean(actor_losses)), float(np.mean(critic_losses)), float(np.mean(entropies))

    def save_checkpoint(self, path: str, episode: int):
        self.model.cpu()
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
        }, path)
        self.model.to(self.device)

    def load_checkpoint(self, checkpoint_path: str):
        import torch

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_dict = self.model.state_dict()
        checkpoint_dict = checkpoint["model_state_dict"]

        filtered_dict = {}
        skipped_keys = []

        for k, v in checkpoint_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    skipped_keys.append(f"{k} (形状不一致: {v.shape} -> {model_dict[k].shape})")
            else:
                skipped_keys.append(f"{k} (現在のモデルに存在しない)")

        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict)

        if skipped_keys:
            print(f"⚠️ 以下の {len(skipped_keys)} 個のパラメータは互換性がないためロードをスキップしました:")
            for key in skipped_keys[:5]:
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  - 他 {len(skipped_keys) - 5} 件...")

        # 🌟 修正: モデル構造に変更があった場合(skipped_keysが存在する場合)は
        #    Optimizer状態はパラメータのindex対応が崩れているため一切復元しない。
        #    中途半端に読み込むと、今回のように shape mismatch が
        #    optimizer.step() 実行時まで表面化しないバグを生むため。
        if skipped_keys:
            print("⚠️ モデル構造の変更を検出したため、Optimizer状態の復元はスキップし、"
                  "オプティマイザは初期状態から再開します。")
        else:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"⚠️ オプティマイザの状態復元に失敗したため、初期状態から再開します: {e}")

        episode = checkpoint.get("episode", 0)
        print(f"チェックポイントの読み込み完了: {checkpoint_path} (episode: {episode})")
        return episode