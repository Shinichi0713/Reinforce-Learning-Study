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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config["model_dir"], run_id)
    os.makedirs(save_path, exist_ok=True)

    agent_ids_onehot = torch.eye(config["num_agents"], device=device)

    # --- 4. 学習ループの開始 ---
    # 初期リセット
    obs_dict, info_dict = env.reset()
    
    # 辞書をテンソルに変換するヘルパー
    def dict_to_tensor(d):
        return torch.FloatTensor(np.array([d[agent] for agent in env.agents])).to(config["device"])

    # 最初の観測と状態をバッファに
    obs_init = dict_to_tensor(obs_dict)
    state_init = obs_init.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)
    buffer.insert_first(obs_init, state_init)

    num_updates = config["total_steps"] // config["episode_length"]
    for update in range(num_updates):
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
            next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(action_dict)
            # print(next_obs_dict)
            
            # データの整形
            next_obs = dict_to_tensor(next_obs_dict)
            next_state = next_obs.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)
            rewards = torch.FloatTensor([[rewards_dict[agent]] for agent in env.agents])
            
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
            train_info = {"value_loss": 0.0} # ログ出力で落ちないように初期化

        # ログと保存
        if update % config["log_interval"] == 0:
            print(f"Update {update}/{num_updates}: Value Loss: {train_info.get('value_loss', 0.0):.4f}")

    env.close()

if __name__ == "__main__":
    train_mappo()