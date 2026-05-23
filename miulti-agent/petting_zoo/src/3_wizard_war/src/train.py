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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- 2. 環境の構築と前処理 ---
    # 1. Parallel APIで初期化
    env = wizard_of_wor_v3.parallel_env(render_mode="rgb_array", auto_rom_install_path=ROM_PATH)
    
    # 2. SuperSuitによる画像の前処理 (重要)
    # env = ss.max_observation_v0(env, 2) # フリッカー対策
    # 修正方法A: 引数名を正しく指定する
    # env = ss.resize_v1(env, x_size=84, y_size=84)

    # # 修正方法B: 位置引数として渡す (こちらの方がシンプルで推奨されます)
    env = ss.max_observation_v0(env, 2)            # フリッカー対策
    env = ss.resize_v1(env, 84, 84)                # 84x84にリサイズ (位置引数で指定)
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
        device=config["device"]
    ).to(config["device"])

    trainer = MAPPOTrainer(ac, config["device"])

    buffer = SharedRolloutBuffer(
        config["num_agents"],
        config["episode_length"],
        obs_shape=obs_shape,
        state_shape=state_shape
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
        # 正しくテンソルに変換
        return torch.FloatTensor(np.array([d[agent] for agent in env.agents])).to(config["device"])

    # 最初の観測と状態をバッファに
    obs_init = dict_to_tensor(obs_dict)
    state_init = obs_init.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)
    buffer.insert_first(obs_init, state_init)

    num_updates = config["total_steps"] // config["episode_length"]
    
    # 報酬とエントロピーの累積用
    episode_rewards = []
    entropy_history = []

    for update in range(start_update, num_updates):
        # エピソードごとの報酬合計
        episode_reward = 0.0
        
        for step in range(config["episode_length"]):
            cur_obs, cur_state = buffer.get_obs_step()
            
            with torch.no_grad():
                # print(cur_obs)
                # print(cur_state)
                # agent_id_onehot を明示的に渡す
                value, action, action_log_prob = ac.get_actions(
                    cur_obs.to(device), 
                    cur_state.to(device),
                    agent_id_onehot=agent_ids_onehot
                )
            
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
                masks = torch.zeros((config["num_agents"], 1), dtype=torch.float32)
            else:
                masks = torch.ones((config["num_agents"], 1), dtype=torch.float32)
            # print(next_obs_dict)
            # データの整形
            next_obs = dict_to_tensor(next_obs_dict)
            # print(f"next_obs shape: {next_obs.shape}")  # デバッグ用

            # next_state を state_shape に合わせて生成
            next_state = next_obs.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)
            # print(f"next_state shape before repeat: {next_state.shape}")  # デバッグ用

            # state_shape が (6, 84, 84) の場合、チャンネル次元を 6 に合わせる
            # 必ず repeat で調整する
            if next_state.size(1) != state_shape[0]:
                print(f"state_shape[0]: {state_shape[0]}, next_state.size(1): {next_state.size(1)}")  # デバッグ用
                # next_state.size(1) が 0 でないことを確認
                if next_state.size(1) == 0:
                    raise ValueError("next_state.size(1) is 0. Check the shape of next_obs.")
                # 安全な整数除算
                repeat_factor = state_shape[0] // next_state.size(1)
                next_state = next_state.repeat(1, repeat_factor, 1, 1)
            else:
                # 一致している場合でも、念のため repeat で調整
                repeat_factor = state_shape[0] // next_state.size(1)
                next_state = next_state.repeat(1, repeat_factor, 1, 1)
            # データの整形
            # next_obs = dict_to_tensor(next_obs_dict)
            # next_state = next_obs.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)
            # rewards = torch.FloatTensor([[rewards_dict[agent]] for agent in env.agers])
            rewards = torch.FloatTensor([[rewards_dict[agent] * 0.01] for agent in env.agents])
            
            # 報酬の累積（エピソード合計）
            episode_reward += rewards.sum().item()

            # 終了判定 (Terminated or Truncated)
            dones = {a: terminations[a] or truncations[a] for a in env.agents}
            masks = torch.FloatTensor([[0.0] if dones[agent] else [1.0] for agent in env.agents])

            # バッファへ保存
            buffer.insert(next_obs, next_state, action, action_log_prob, value, rewards, masks)

            # エピソード終了時のリセット
            if any(dones.values()):
                obs_dict, _ = env.reset()
                # print(obs_dict)
                # (注) ここでバッファの整合性を保つための処理が必要な場合があります
        # print("hi")
        # 報酬の計算と更新
        with torch.no_grad():
            last_state = buffer.state[-1].to(config["device"]).float()
            # 変数名を agent_ids_onehot (複数形) に修正
            next_value = ac.get_value(last_state, agent_ids_onehot) 
            buffer.compute_returns(next_value, gamma=0.99, gae_lambda=0.95)       
        
        train_info = trainer.train(buffer)
        buffer.after_update()

        # ログと保存
        # trainer.train が None を返した場合のフォールバック
        if train_info is None:
            print(f"Update {update}: trainer.train が値を返しませんでした。")
            train_info = {"value_loss": 0.0, "entropy": 0.0} # ログ出力で落ちないように初期化

        # 報酬とエントロピーの記録
        episode_rewards.append(episode_reward)
        entropy_history.append(train_info.get('entropy', 0.0))

        # ログと保存
        if update % config["log_interval"] == 0:
            # 平均報酬と平均エントロピーを計算
            avg_reward = np.mean(episode_rewards[-config["log_interval"]:])
            avg_entropy = np.mean(entropy_history[-config["log_interval"]:])
            print(f"Update {update}/{num_updates}: "
                  f"Value Loss: {train_info.get('value_loss', 0.0):.4f}, "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Avg Entropy: {avg_entropy:.4f}")

        # --- モデルの保存 ---
        if update % config["save_interval"] == 0:
            model_path = os.path.join(CHECKPOINT_DIR, f"mappo_model.pth")
            torch.save({
                'model_state_dict': ac.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'update': update,
                'config': config
            }, model_path)
            print(f"Model saved to {model_path}")

    env.close()

if __name__ == "__main__":
    train_mappo()