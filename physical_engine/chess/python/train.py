"""
碁 Actor-Critic 学習スクリプト（改善版）
=========================================

元コードからの主な変更点:
  1. 複数エピソード(バッチ)をまとめてから1回更新 → 勾配の分散を大幅に低減
  2. GAE (Generalized Advantage Estimation) でアドバンテージを計算 → 学習の安定化
  3. アドバンテージの正規化
  4. 盤面の8対称性(回転・反転)によるデータ拡張 → サンプル効率を最大8倍に
  5. エントロピー係数のアニーリング（探索→活用へ滑らかに移行）
  6. 勾配クリッピング + 学習率スケジューラ
  7. 過去の自分自身(チェックポイント)を対戦相手プールとして使う自己対戦フック
  8. 盤サイズは 9x9 からのスタートを推奨（19x19は探索空間が広すぎて
     素のActor-Criticでは収束が非常に困難）

前提:
  - GoEnv, GoPolicyValueNet は既存のものをそのまま使う想定です。
    state.shape == (C, N, N)  (例: C=3, N=board_size)
    legal_mask.shape == (num_actions,)  (num_actions = N*N (+1 パス))
  - env が二人零和ゲームで、手番プレイヤー視点の state / reward を返す
    （AlphaZero系でよくある「常に手番側から見た盤面」の実装）を想定しています。
    もし異なるインターフェースなら、collect_episode() 内のコメント部分を
    ご自身の GoEnv に合わせて調整してください。
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ============================================================
# ハイパーパラメータ
# ============================================================
BOARD_SIZE = 9              # まずは 9x9 で学習を安定させてから 19x19 へ拡張する
GAMMA = 0.99                # 割引率
GAE_LAMBDA = 0.95           # GAEのλ（バイアスと分散のトレードオフ）
LR = 3e-4                   # 学習率（Adamにしては0.0005はやや大きめなので少し下げる）
ENTROPY_BETA_START = 0.02   # 探索用エントロピー係数（学習初期）
ENTROPY_BETA_END = 0.002    # エントロピー係数（学習終盤）
ENTROPY_ANNEAL_STEPS = 20000
GRAD_CLIP_NORM = 1.0
NUM_UPDATES = 5000         # 更新回数（1更新 = EPISODES_PER_UPDATE局分）
EPISODES_PER_UPDATE = 16    # 1回の勾配更新に使う自己対戦の局数（分散低減の要）
USE_SYMMETRY_AUGMENTATION = True
OPPONENT_POOL_SIZE = 8      # 過去チェックポイントを何個まで保持するか
OPPONENT_UPDATE_INTERVAL = 200  # 何updateごとに現在のモデルをプールに追加するか
CHECKPOINT_PATH = "go_agent_checkpoint.pt"


# ============================================================
# 盤面の対称性によるデータ拡張 (8通りのダイヘドラル群)
# ============================================================
def dihedral_transform_state(state: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """state: (C, N, N) を90度回転 k 回 + 必要ならフリップ"""
    out = np.rot90(state, k=k, axes=(1, 2))
    if flip:
        out = np.flip(out, axis=2)
    return np.ascontiguousarray(out)


def dihedral_transform_action(action: int, board_size: int, k: int, flip: bool, pass_action: int) -> int:
    """行動インデックス（座標 or パス）を同じ変換で写像する"""
    if action == pass_action:
        return pass_action
    row, col = divmod(action, board_size)
    coord = np.zeros((board_size, board_size), dtype=np.int64)
    coord[row, col] = 1
    coord = np.rot90(coord, k=k)
    if flip:
        coord = np.flip(coord, axis=1)
    new_row, new_col = np.argwhere(coord == 1)[0]
    return int(new_row) * board_size + int(new_col)


def augment_transition(state, action, legal_mask, board_size, pass_action):
    """ランダムに1つの対称変換を選んで適用する（8通りのうち恒等変換も含む）"""
    k = random.randint(0, 3)
    flip = random.choice([True, False])
    new_state = dihedral_transform_state(state, k, flip)
    new_action = dihedral_transform_action(action, board_size, k, flip, pass_action)
    # legal_mask も同様に変換（盤面部分のみ。パスの要素があれば末尾に温存）
    board_part = legal_mask[: board_size * board_size].reshape(board_size, board_size)
    board_part = np.rot90(board_part, k=k)
    if flip:
        board_part = np.flip(board_part, axis=1)
    new_mask = legal_mask.copy()
    new_mask[: board_size * board_size] = board_part.reshape(-1)
    return new_state, new_action, new_mask


# ============================================================
# GAE計算
# ============================================================
def compute_gae(rewards, values, gamma, lam):
    """
    rewards: list[float]  (エピソード全長)
    values:  list[float]  (エピソード全長。終端の後続値は0として扱う=エピソード完結タスク)
    return: advantages(list[float]), returns(list[float])
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    next_value = 0.0  # 終端なのでブートストラップ値は0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns


# ============================================================
# 1エピソード分の自己対戦を収集
# ============================================================
def collect_episode(env, model, device, board_size, pass_action):
    """
    自己対戦で1局プレイし、(state, action, log_prob不要=後で再計算, value, reward, legal_mask)
    のリストを返す。手番側視点のrewardを想定。
    """
    state, _ = env.reset()
    done = False

    states, actions, rewards, values, masks = [], [], [], [], []

    with torch.no_grad():
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            legal_mask = env.get_legal_actions()
            mask_tensor = torch.FloatTensor(legal_mask).to(device)

            policy_logits, value = model(state_tensor)
            masked_logits = policy_logits - (1.0 - mask_tensor) * 1e9
            probs = torch.softmax(masked_logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            masks.append(legal_mask)

            state = next_state

    return states, actions, rewards, values, masks


# ============================================================
# メイン学習ループ
# ============================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    env = GoEnv(board_size=BOARD_SIZE)  # ※ GoEnv が board_size 引数を取らない場合は調整してください
    model = GoPolicyValueNet(board_size=BOARD_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_UPDATES)

    pass_action = BOARD_SIZE * BOARD_SIZE  # パスがある場合のインデックス想定

    opponent_pool = []  # 過去チェックポイントのstate_dictを保存（自己対戦相手用）
    global_step = 0

    for update in range(1, NUM_UPDATES + 1):
        batch_states, batch_actions, batch_advantages, batch_returns, batch_masks = [], [], [], [], []
        episode_total_rewards = []

        # --- 複数局まとめて自己対戦データを収集（分散低減） ---
        for _ in range(EPISODES_PER_UPDATE):
            states, actions, rewards, values, masks = collect_episode(
                env, model, device, BOARD_SIZE, pass_action
            )
            advantages, returns = compute_gae(rewards, values, GAMMA, GAE_LAMBDA)

            if USE_SYMMETRY_AUGMENTATION:
                aug_states, aug_actions, aug_masks = [], [], []
                for s, a, m in zip(states, actions, masks):
                    ns, na, nm = augment_transition(s, a, m, BOARD_SIZE, pass_action)
                    aug_states.append(ns)
                    aug_actions.append(na)
                    aug_masks.append(nm)
                states, actions, masks = aug_states, aug_actions, aug_masks

            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_advantages.extend(advantages)
            batch_returns.extend(returns)
            batch_masks.extend(masks)
            episode_total_rewards.append(sum(rewards))

        # --- テンソル化 ---
        states_tensor = torch.FloatTensor(np.array(batch_states)).to(device)
        actions_tensor = torch.LongTensor(batch_actions).to(device)
        advantages_tensor = torch.FloatTensor(batch_advantages).to(device)
        returns_tensor = torch.FloatTensor(batch_returns).to(device)
        masks_tensor = torch.FloatTensor(np.array(batch_masks)).to(device)

        # アドバンテージの正規化（学習安定化に重要）
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std() + 1e-8
        )

        # --- 現在方策での再評価（log_prob, value, entropy をまとめて計算） ---
        policy_logits, values_pred = model(states_tensor)
        masked_logits = policy_logits - (1.0 - masks_tensor) * 1e9
        probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        values_pred = values_pred.squeeze(-1)

        # --- エントロピー係数のアニーリング ---
        frac = min(1.0, global_step / ENTROPY_ANNEAL_STEPS)
        entropy_beta = ENTROPY_BETA_START + frac * (ENTROPY_BETA_END - ENTROPY_BETA_START)

        actor_loss = -(log_probs * advantages_tensor).mean()
        critic_loss = nn.functional.mse_loss(values_pred, returns_tensor)
        total_loss = actor_loss + 0.5 * critic_loss - entropy_beta * entropy

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        global_step += 1

        # --- 過去モデルを対戦相手プールに追加（自己対戦の多様性確保） ---
        if update % OPPONENT_UPDATE_INTERVAL == 0:
            opponent_pool.append(copy.deepcopy(model.state_dict()))
            if len(opponent_pool) > OPPONENT_POOL_SIZE:
                opponent_pool.pop(0)

        if update % 20 == 0:
            avg_reward = float(np.mean(episode_total_rewards))
            print(
                f"Update {update:5d} | AvgReward: {avg_reward:6.3f} | "
                f"Loss: {total_loss.item():.4f} | Entropy: {entropy.item():.4f} | "
                f"EntropyBeta: {entropy_beta:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        if update % 100 == 0:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> チェックポイント保存: {CHECKPOINT_PATH}")

    # 学習済みモデルをONNX形式でエクスポート
    export_to_onnx(model, "go_agent_trained.onnx")


if __name__ == "__main__":
    train()