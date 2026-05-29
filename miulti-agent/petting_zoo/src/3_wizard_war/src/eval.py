import os
import torch
import numpy as np
import cv2
from datetime import datetime
from pettingzoo.atari import wizard_of_wor_v3
import supersuit as ss

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

def record_episode(model_path, output_dir="./videos", episode_length=1000):
    """
    学習済みモデルで1エピソードを再生し、動画として保存する

    Args:
        model_path (str): 学習済みモデルのパス
        output_dir (str): 動画の保存ディレクトリ
        episode_length (int): エピソードの最大ステップ数
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # モデルのロード
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint['config']

    # 環境の構築（学習時と同じ前処理）
    env = wizard_of_wor_v3.parallel_env(render_mode="rgb_array", auto_rom_install_path=ROM_PATH)
    env = ss.max_observation_v0(env, 2)
    env = ss.resize_v1(env, 84, 84)
    env = ss.dtype_v0(env, 'float32')
    env = ss.normalize_obs_v0(env, 0, 255)
    env = ss.observation_lambda_v0(env, lambda obs, obs_space: np.transpose(obs, (2, 0, 1)))

    # 形状の定義
    obs_shape = (3, 84, 84)
    state_shape = (obs_shape[0] * config["num_agents"], 84, 84)
    action_n = env.action_space(env.possible_agents[0]).n

    # モデルの初期化
    ac = MAPPO_ActorCritic(
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_space_n=action_n,
        num_agents=config["num_agents"],
        device="cpu"
    )
    ac.load_state_dict(checkpoint['model_state_dict'])
    ac.eval()

    # Agent ID One-hot
    agent_ids_onehot = torch.eye(config["num_agents"], device="cpu")

    # 動画の設定
    video_path = os.path.join(output_dir, f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30

    # 環境リセット
    obs_dict, _ = env.reset()

    # 最初のフレームを取得して動画のサイズを決定
    first_frame = env.render()
    height, width, _ = first_frame.shape
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # フレームを書き込み
    video_writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    # エピソードの実行
    total_reward = 0.0
    step = 0

    while step < episode_length:
        # 観測をテンソルに変換
        obs = torch.FloatTensor(np.array([obs_dict[agent] for agent in env.agents])).to("cpu")
        state = obs.view(-1, 84, 84).repeat(config["num_agents"], 1, 1, 1)

        # 行動の決定
        with torch.no_grad():
            _, action, _ = ac.get_actions(obs, state, agent_id_onehot=agent_ids_onehot)

        # 行動を辞書に変換
        action_dict = {agent: action[i].item() for i, agent in enumerate(env.agents)}

        # 環境のステップ実行
        next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(action_dict)

        # 報酬の合計
        total_reward += sum(rewards_dict.values())

        # フレームを取得して動画に書き込み
        frame = env.render()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # エピソード終了判定
        if any(terminations.values()) or any(truncations.values()):
            break

        # 次の観測に更新
        obs_dict = next_obs_dict
        step += 1

    # 動画の保存
    video_writer.release()
    env.close()

    print(f"Episode recorded to {video_path}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps: {step}")

    return video_path, total_reward, step

# 使用例
if __name__ == "__main__":
    # 最新のチェックポイントを検索
    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"Recording episode with model: {latest_checkpoint}")
        video_path, total_reward, steps = record_episode(latest_checkpoint)
    else:
        print("No checkpoint found. Please train the model first.")