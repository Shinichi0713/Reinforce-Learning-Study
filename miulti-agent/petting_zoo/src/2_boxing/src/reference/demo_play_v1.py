import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

def play_boxing(agent_path, num_episodes=1, render=True):
    """
    保存したモデルを読み込んでボクシングをプレイさせる
    """
    # 1. 環境の準備
    env = get_env() # 以前定義した関数を使用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. エージェントの準備とロード
    agent = MAPPOAgent(action_space_n=18)
    print(f"Loading model from: {agent_path}")
    checkpoint = torch.load(agent_path, map_location=device)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.to(device)
    agent.eval() # 推論モードに設定

    for episode in range(num_episodes):
        obs_dict, _ = env.reset()
        done = False
        total_rewards = {'first_0': 0, 'second_0': 0}
        
        print(f"--- Episode {episode + 1} Start ---")

        while not done:
            # 前処理 (チャンネル順序の変換と正規化)
            o1, o2, _ = preprocess_joint_obs(obs_dict, device)

            with torch.no_grad():
                # テスト時は確率が最大のものを選ぶ（決定論的）
                # 1Pの行動
                features1 = agent.actor_encoder(o1.unsqueeze(0))
                logits1 = agent.action_head(features1)
                a1 = torch.argmax(logits1, dim=-1).item()

                # 2Pの行動
                features2 = agent.actor_encoder(o2.unsqueeze(0))
                logits2 = agent.action_head(features2)
                a2 = torch.argmax(logits2, dim=-1).item()

            # 環境を進める
            actions = {'first_0': a1, 'second_0': a2}
            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            
            # 報酬の加算
            for k in rewards:
                total_rewards[k] += rewards[k]

            # 描画 (Google Colab用)
            if render:
                display.clear_output(wait=True)
                # raw_envのrenderを使用して画面を表示
                img = env.render()
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Episode: {episode+1} | Score: 1P {total_rewards['first_0']} - 2P {total_rewards['second_0']}")
                plt.show()

            done = any(terms.values()) or any(truncs.values())

        print(f"Episode {episode + 1} Finished. Total Score: {total_rewards}")

# --- 実行 ---
# Googleドライブの最新モデルを指定
MODEL_PATH = "/content/drive/MyDrive/rl_boxing/checkpoints/mappo_agent_latest.pth"
play_boxing(MODEL_PATH)