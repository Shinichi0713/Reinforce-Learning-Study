"""
Multi-Agent Transformer (MAT) 版の Actor-Critic 実装。
既存の MAPPO_TransformerActor / MAPPO_TransformerCritic (元コード) を土台に、
「エージェント間の自己回帰的デコード」を追加したもの。

参考: Wen et al., "Multi-Agent Reinforcement Learning is a Sequence Modeling
      Problem", NeurIPS 2022.

設計方針:
  1. MATObsEncoder  : 元コードの空間Transformerをそのまま流用し、
                       1エージェント分の局所観測(236次元)をd_model次元に要約する。
  2. MATEncoder      : 8体分の要約特徴を1つの系列とみなし、エージェント間で
                       自己注意させて相互に文脈化する(元Criticの役割に相当)。
                       各エージェント自身の文脈化表現から直接Valueを出す。
  3. MATDecoder      : エージェント0->7の順に、直前エージェントの実際の行動を
                       条件として次のエージェントの行動分布を自己回帰的に予測する。
                       LLMの「1トークンずつ生成」と全く同じ構造(teacher forcing /
                       autoregressive decoding)。

NOTE: 実行環境にネットワークがなくPyTorchをインストールできないため、
      このファイルは構文チェック(py_compile)のみ実施し、実行検証はしていません。
      ご自身の環境で最終確認してください。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class MATObsEncoder(nn.Module):
    """
    元コード MAPPO_TransformerActor の空間Transformer部分を流用。
    1エージェント分の局所観測 (obs_range^2 * in_channels + num_agents*5 次元) を
    d_model 次元の要約ベクトルに変換する。
    """

    def __init__(self, obs_range=7, in_channels=4, d_model=64, nhead=4,
                 num_layers=2, num_agents=8):
        super().__init__()
        self.obs_range = obs_range
        self.num_tokens = obs_range * obs_range          # 7x7 = 49
        self.spatial_dim = obs_range * obs_range * in_channels  # 196
        self.num_agents = num_agents

        # 中心マス(自分の位置)のフラットインデックスを明示的に計算 (obs_range=7 -> 24)
        c = obs_range // 2
        self.center_idx = c * obs_range + c

        self.embedding = nn.Linear(in_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        self.action_history_embed = nn.Sequential(
            nn.Linear(num_agents * 5, d_model),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, spatial_dim + num_agents*5)  1エージェント分のフラット観測
        Returns: (B, d_model)  自分の位置に対応する要約特徴
        """
        B = obs.shape[0]
        spatial_obs = obs[:, :self.spatial_dim]
        action_history = obs[:, self.spatial_dim:]

        x = spatial_obs.view(B, self.num_tokens, -1)
        x = self.embedding(x) + self.pos_embedding

        act_emb = self.action_history_embed(action_history).unsqueeze(1)
        x = x + act_emb

        features = self.spatial_transformer(x)
        my_feature = features[:, self.center_idx, :]
        return my_feature


class MATEncoder(nn.Module):
    """
    8体分の要約特徴を1つの系列とみなし、エージェント間で自己注意させる。
    元コードの MAPPO_TransformerCritic が「392トークンを1つのシーケンスとして
    自己注意」させていたのと同じ発想だが、こちらは各エージェント単位(8トークン)で
    行い、各エージェント自身の文脈化された表現からValueを直接算出する。
    """

    def __init__(self, num_agents=8, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.num_agents = num_agents
        self.agent_pos_embedding = nn.Parameter(torch.randn(1, num_agents, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, agent_feats: torch.Tensor):
        """
        agent_feats: (B, num_agents, d_model)  MATObsEncoder の出力を8体分束ねたもの
        Returns:
            enc_out: (B, num_agents, d_model)  相互に文脈化されたエージェント表現
            values:  (B, num_agents)           各エージェントの価値推定
        """
        x = agent_feats + self.agent_pos_embedding
        enc_out = self.transformer(x)
        values = self.value_head(enc_out).squeeze(-1)
        return enc_out, values


class MATDecoder(nn.Module):
    """
    エージェント0->7の順に、直前エージェントの実際の行動を条件として
    次のエージェントの行動分布を自己回帰的に予測する。
    """

    def __init__(self, num_agents=8, action_dim=5, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.START = action_dim  # 開始トークンのID (0..action_dim-1が実際の行動)

        self.action_embedding = nn.Embedding(action_dim + 1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_agents, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(d_model, action_dim)

        # 自分より後のエージェントの行動を参照しないための因果マスク
        causal_mask = torch.triu(
            torch.full((num_agents, num_agents), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, enc_out: torch.Tensor, shifted_actions: torch.Tensor) -> torch.Tensor:
        """
        教師強制(teacher forcing)による学習時のフォワード。

        enc_out: (B, num_agents, d_model)          MATEncoder の出力
        shifted_actions: (B, num_agents) long       [START, a_0, a_1, ..., a_6] のように
                                                     実際の行動系列を1つ右にシフトしたもの
        Returns:
            logits: (B, num_agents, action_dim)     各エージェント位置の行動ロジット
        """
        tgt = self.action_embedding(shifted_actions) + self.pos_embedding
        dec_out = self.transformer_decoder(tgt=tgt, memory=enc_out, tgt_mask=self.causal_mask)
        logits = self.action_head(dec_out)
        return logits

    @torch.no_grad()
    def autoregressive_decode(self, enc_out: torch.Tensor, greedy: bool = False):
        """
        ロールアウト収集時: エージェント0から7まで実際に1体ずつサンプルしながら生成する。

        enc_out: (B, num_agents, d_model)
        Returns:
            actions:   (B, num_agents) long
            log_probs: (B, num_agents)
        """
        B = enc_out.shape[0]
        device = enc_out.device

        current_input = torch.full((B, self.num_agents), self.START, dtype=torch.long, device=device)
        actions = torch.zeros((B, self.num_agents), dtype=torch.long, device=device)
        log_probs = torch.zeros((B, self.num_agents), device=device)

        for i in range(self.num_agents):
            tgt = self.action_embedding(current_input) + self.pos_embedding
            dec_out = self.transformer_decoder(tgt=tgt, memory=enc_out, tgt_mask=self.causal_mask)
            logits_i = self.action_head(dec_out[:, i, :])
            dist_i = Categorical(logits=logits_i)

            a_i = torch.argmax(logits_i, dim=-1) if greedy else dist_i.sample()

            actions[:, i] = a_i
            log_probs[:, i] = dist_i.log_prob(a_i)

            if i + 1 < self.num_agents:
                current_input[:, i + 1] = a_i

        return actions, log_probs


class MATActorCritic(nn.Module):
    """MATObsEncoder + MATEncoder + MATDecoder をまとめたトップレベルモジュール。"""

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

    def encode(self, joint_obs: torch.Tensor):
        """joint_obs: (B, num_agents, obs_dim) -> enc_out, values"""
        B, N, D = joint_obs.shape
        flat_obs = joint_obs.reshape(B * N, D)
        agent_feats = self.obs_encoder(flat_obs).view(B, N, -1)
        enc_out, values = self.encoder(agent_feats)
        return enc_out, values

    def forward_train(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor):
        """
        学習(PPO update)時: 教師強制でログ確率・エントロピー・価値を一括計算する。

        joint_obs:     (B, num_agents, obs_dim)
        joint_actions: (B, num_agents) long  ロールアウト時に実際に取った行動
        Returns:
            log_probs: (B, num_agents)
            entropy:   (B, num_agents)
            values:    (B, num_agents)
        """
        enc_out, values = self.encode(joint_obs)

        B = joint_obs.shape[0]
        start_col = torch.full((B, 1), self.decoder.START, dtype=torch.long, device=joint_obs.device)
        shifted_actions = torch.cat([start_col, joint_actions[:, :-1]], dim=1)

        logits = self.decoder(enc_out, shifted_actions)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(joint_actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    @torch.no_grad()
    def act(self, joint_obs: torch.Tensor, greedy: bool = False):
        """
        ロールアウト収集時: エージェント0->7の順に自己回帰的に行動をサンプルする。

        joint_obs: (B, num_agents, obs_dim)
        Returns:
            actions:   (B, num_agents) long
            log_probs: (B, num_agents)
            values:    (B, num_agents)
        """
        enc_out, values = self.encode(joint_obs)
        actions, log_probs = self.decoder.autoregressive_decode(enc_out, greedy=greedy)
        return actions, log_probs, values
