import os
import torch
import numpy as np
from datetime import datetime
from pettingzoo.atari import wizard_of_wor_v3
import supersuit as ss  # 前処理用に必要です

# ※ クラス定義はインポートされている前提です
# from model import MAPPO_ActorCritic
# from buffer import SharedRolloutBuffer
# from trainer import MAPPOTrainer
ROM_PATH = "/usr/local/lib/python3.12/dist-packages/AutoROM/roms/"

def compute_rewards(rewards_dict, terminations, truncations, num_agents, device, shot_this_step):
    """
    報酬の計算（死亡ペナルティ + 生存報酬 + 味方撃ちペナルティ + スケーリング）

    Args:
        rewards_dict (dict): env.step() で返される報酬辞書
        terminations (dict): 終了フラグ
        truncations (dict): 打ち切りフラグ
        num_agents (int): エージェント数
        device (torch.device): デバイス
        shot_this_step (dict): 各エージェントがこのステップで弾を撃ったか

    Returns:
        torch.FloatTensor: 形状 (num_agents, 1) の報酬テンソル
    """
    modified_rewards = rewards_dict.copy()

    for agent in modified_rewards.keys():
        base_reward = modified_rewards[agent]

        # 生存時間報酬
        survival_reward = 0.1

        # 死亡ペナルティ
        death_penalty = 0.0
        if terminations.get(agent, False) or truncations.get(agent, False):
            death_penalty = -50.0

        # ★ 味方撃ちペナルティ
        friendly_fire_penalty = 0.0
        if (terminations.get(agent, False) or truncations.get(agent, False)) and shot_this_step[agent]:
            # 死んだ かつ 直前で弾を撃った → 味方を撃った可能性が高い
            friendly_fire_penalty = -50.0  # 例: 味方撃ちペナルティ

        # 合計報酬
        modified_rewards[agent] = (
            base_reward + survival_reward + death_penalty + friendly_fire_penalty
        )

    # スケーリング
    rewards = torch.FloatTensor(
        [[modified_rewards[agent] * 0.01] for agent in modified_rewards.keys()]
    ).to(device)
    return rewards


def find_latest_checkpoint(checkpoint_dir):
    """最新のチェックポイントファイルを検索"""
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("mappo_model") and f.endswith(".pth")]
    if len(files) == 1:
      return os.path.join(checkpoint_dir, files[0])
    else:
      if not files:
          return None
      # update番号でソート
      files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

      print(files)
      return os.path.join(checkpoint_dir, files[-1])

def train_mappo():
    # --- 1. ハイパーパラメータ・設定 ---
    config = {
        "env_name": "WizardOfWor-v3",
        "num_agents": 2,
        "total_steps": 10000000,
        "episode_length": 200,
        "save_interval": 50,
        "log_interval": 10,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "model_dir": "./models"
    }
    # ★ 修正: device を一箇所で定義し、以降はこれを使う
    device = config["device"]

    # --- 2. 環境の構築と前処理 ---
    # 1. Parallel APIで初期化
    env = wizard_of_wor_v3.parallel_env(render_mode="rgb_array", auto_rom_install_path=ROM_PATH)

    # 2. SuperSuitによる画像の前処理 (重要)
    env = ss.max_observation_v0(env, 2)            # フリッカー対策
    env = ss.resize_v1(env, 84, 84)                # 84x84にリサイズ
    env = ss.dtype_v0(env, 'float32')              # float32に変換
    env = ss.normalize_obs_v0(env, 0, 255)         # 0.0-1.0に正規化

    # 関数名に依存しない汎用的な次元入れ替え (H, W, C) -> (C, H, W)
    env = ss.observation_lambda_v0(env, lambda obs, obs_space: np.transpose(obs, (2, 0, 1)))

    # 形状の定義
    obs_shape = (3, 84, 84)
    # 集中状態は全エージェントの観測を結合 (3ch * 2人 = 6ch)
    state_shape = (obs_shape[0] * config["num_agents"], 84, 84)

    # 行動数の取得
    example_agent = env.possible_agents[0]
    action_n = env.action_space(example_agent).n

    # --- 3. コンポーネントの初期化 ---
    # モデルのインスタンス化 (前回のエラーを防止)
    ac = MAPPO_ActorCritic(
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_space_n=action_n,
        num_agents=config["num_agents"],
        device=device  # ★ 修正: config["device"] ではなく device を使う
    ).to(device)

    trainer = MAPPOTrainer(ac, device)

    buffer = SharedRolloutBuffer(
        config["num_agents"],
        config["episode_length"],
        obs_shape=obs_shape,
        state_shape=state_shape,
        device=device  # ★ 修正: buffer にも device を渡す（SharedRolloutBuffer 側でテンソルを device に乗せる想定）
    )

    # 保存ディレクトリ
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- チェックポイントのロード ---
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    start_update = 0
    if latest_checkpoint:
        print(f"Loading checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        ac.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_update = checkpoint.get('update', 0) + 1  # 次の更新から再開
        print(f"Resuming from update {start_update}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # ★ 修正: agent_ids_onehot も device に合わせる
    agent_ids_onehot = torch.eye(config["num_agents"], device=device)

    # --- 4. 学習ループの開始 ---
    # 初期リセット
    obs_dict, info_dict = env.reset()

    # 辞書をテンソルに変換するヘルパー
    def dict_to_tensor(d):
        # env.agents が空の場合の処理
        if not env.agents:
            # 環境をリセットして再試行
            d, _ = env.reset()
        # ★ 修正: テンソルを device に乗せる
        return torch.FloatTensor(np.array([d[agent] for agent in env.agents])).to(device)

    # 最初の観測と状態をバッファに
    obs_init = dict_to_tensor(obs_dict)
    state_init = obs_init.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)
    buffer.insert_first(obs_init.to(device), state_init.to(device))  # ★ 追加
    num_updates = config["total_steps"] // config["episode_length"]

    # 報酬とエントロピーの累積用
    episode_rewards = []
    entropy_history = []
    # 弾発射の記録用
    shot_this_step = {agent: False for agent in env.possible_agents}
    for update in range(start_update, num_updates):
        # エピソードごとの報酬合計
        episode_reward = 0.0

        for step in range(config["episode_length"]):
            cur_obs, cur_state = buffer.get_obs_step()
            # ★ 修正: buffer がすでに device に合わせている前提なので、ここで .to(device) は不要
            #         もし buffer 側で device 指定がない場合は、ここで .to(device) を追加
            cur_obs = cur_obs.to(device)
            cur_state = cur_state.to(device)

            with torch.no_grad():
                # agent_id_onehot を明示的に渡す
                value, action, action_log_prob = ac.get_actions(
                    cur_obs,  # ★ 修正: .to(device) を削除
                    cur_state,  # ★ 修正: .to(device) を削除
                    agent_id_onehot=agent_ids_onehot
                )
            # ★ 弾発射の記録
            for i, agent in enumerate(env.agents):
                # 例: 行動番号 1 が「弾を撃つ」行動と仮定（環境の仕様に合わせて変更）
                if action[i].item() == 1:  # 弾発射行動
                    shot_this_step[agent] = True
                else:
                    shot_this_step[agent] = False

            # env.step 用に行動を辞書に変換
            action_dict = {agent: action[i].item() for i, agent in enumerate(env.agents)}

            # 環境の実行
            agents_before = set(env.agents)
            next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(action_dict)
            agents_after = set(env.agents)

            dead_agents = agents_before - agents_after
            if dead_agents:
                print(f"Dead agents: {dead_agents}")
            # エピソード終了時の処理
            if any(terminations.values()) or any(truncations.values()):
                # 環境をリセット
                next_obs_dict, _ = env.reset()
                # マスクを 0 に設定（終了時）
                masks = torch.zeros((config["num_agents"], 1), dtype=torch.float32, device=device)  # ★ 修正: device 指定
            else:
                masks = torch.ones((config["num_agents"], 1), dtype=torch.float32, device=device)  # ★ 修正: device 指定

            # データの整形
            next_obs = dict_to_tensor(next_obs_dict)

            # next_state を state_shape に合わせて生成
            next_state = next_obs.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)

            # state_shape が (6, 84, 84) の場合、チャンネル次元を 6 に合わせる
            if next_state.size(1) != state_shape[0]:
                if next_state.size(1) == 0:
                    raise ValueError("next_state.size(1) is 0. Check the shape of next_obs.")
                repeat_factor = state_shape[0] // next_state.size(1)
                next_state = next_state.repeat(1, repeat_factor, 1, 1)
            else:
                repeat_factor = state_shape[0] // next_state.size(1)
                next_state = next_state.repeat(1, repeat_factor, 1, 1)

            # --- ここで compute_rewards を呼び出して報酬を計算 ---
            rewards = compute_rewards(
                rewards_dict=rewards_dict,
                terminations=terminations,
                truncations=truncations,
                num_agents=config["num_agents"],
                device=device,  # ★ 修正: device を渡す
                shot_this_step=shot_this_step  # ★ 追加
            )

            # 報酬の累積（エピソード合計）
            episode_reward += rewards.sum().item()

            # 終了判定 (Terminated or Truncated)
            dones = {a: terminations[a] or truncations[a] for a in env.agents}
            # ★ 修正: masks はすでに上で device に合わせて生成しているので、ここで再生成する必要はない
            #         もし必要なら device を指定
            # masks = torch.FloatTensor([[0.0] if dones[agent] else [1.0] for agent in env.agents]).to(device)

            # バッファへ保存
            buffer.insert(
                next_obs.to(device),        # ★ 追加
                next_state.to(device),      # ★ 追加
                action,
                action_log_prob,
                value,
                rewards,
                masks
            )

            # エピソード終了時のリセット
            if any(dones.values()):
                obs_dict, _ = env.reset()

        # 報酬の計算と更新
        with torch.no_grad():
            last_state = buffer.state[-1].to(device).float()
            next_value = ac.get_value(last_state, agent_ids_onehot)
            buffer.compute_returns(next_value, gamma=0.99, gae_lambda=0.95)

        train_info = trainer.train(buffer)
        buffer.after_update()

        # ログと保存
        if train_info is None:
            print(f"Update {update}: trainer.train が値を返しませんでした。")
            train_info = {"value_loss": 0.0, "entropy": 0.0}

        # 報酬とエントロピーの記録
        episode_rewards.append(episode_reward)
        entropy_history.append(train_info.get('entropy', 0.0))

        # ログと保存
        if update % config["log_interval"] == 0:
            avg_reward = np.mean(episode_rewards[-config["log_interval"]:])
            avg_entropy = np.mean(entropy_history[-config["log_interval"]:])
            print(f"Update {update}/{num_updates}: "
                  f"Value Loss: {train_info.get('value_loss', 0.0):.4f}, "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Avg Entropy: {avg_entropy:.4f}")

        # --- モデルの保存 ---
        if update % config["save_interval"] == 0:
            model_path = os.path.join(CHECKPOINT_DIR, f"mappo_model.pth")
            ac.cpu()
            torch.save({
                'model_state_dict': ac.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'update': update,
                'config': config
            }, model_path)
            ac.to(device)
            print(f"Model saved to {model_path}")

    env.close()

if __name__ == "__main__":
    train_mappo()