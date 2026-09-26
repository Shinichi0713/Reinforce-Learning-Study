
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env import MultiSensorSearchEnv
from trainer import SharedAgent
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def load_trained_model(file_path: str, grid_size: int, action_space: int, map_location=None):
    """
    指定されたファイルパスから学習済みモデルと履歴をロードする。

    Args:
        file_path (str): モデルの保存ファイルパス (例: 'final_exploration_model_10x10.pth')
        grid_size (int): ロードする環境のグリッドサイズ (モデル構造の再構築に必要)
        action_space (int): ロードする環境のアクション数 (モデル構造の再構築に必要)
        map_location: モデルをロードするデバイスを指定 (例: 'cpu' または None)

    Returns:
        tuple: (policy_net, optimizer, saved_data)
    """
    if not os.path.exists(file_path):
        print(f"エラー: 指定されたファイル {file_path} が見つかりません。")
        return None, None, None

    # 1. 保存データのロード
    # map_locationを設定することで、GPUで保存されたモデルをCPU環境でもロード可能
    saved_data = torch.load(file_path, map_location=map_location)
    print(f"✅ モデルファイルをロードしました。エピソード数: {saved_data['episode']}")

    # 2. モデルの再構築 (新しいインスタンスの作成)
    # DuelingDQNの構造を再構築するために、保存されたハイパーパラメータを使っても良い
    policy_net = SharedAgent(grid_size, action_space)
    # optimizer = optim.Adam(policy_net.parameters(), lr=0.001) # LRは学習時と同じである必要あり

    # 3. state_dictの適用
    # モデルの重みをインスタンスに適用
    policy_net.policy_net.load_state_dict(saved_data['model_state_dict'])
    
    # 4. オプティマイザの状態の適用 (学習を継続する場合に必須)
    # optimizer.load_state_dict(saved_data['optimizer_state_dict'])
        
    return policy_net

# --- 学習済みモデルでのテスト実行 ---
print("\n--- テスト実行 (可視化) ---")
env = MultiSensorSearchEnv(size=10, num_agents=3)
obs = env.reset()
trained_brain = load_trained_model(f"{os.path.dirname(__file__)}/trained_search_agent.pth", 10, 5, map_location=device)
state = {i: trained_brain.preprocess_state(obs, i) for i in range(env.num_agents)}

# 探索率(ε)を0にして、学習した知識のみで動かす
trained_brain.policy_net.eval()

for t in range(100):
    actions = {}
    for i in range(env.num_agents):
        with torch.no_grad():
            # ε-greedyを使わず、最大のQ値を持つ行動を選択
            action = trained_brain.policy_net(state[i]).max(1)[1].item()
        actions[i] = action
    print(actions)
    next_obs, rewards, done, info = env.step(actions)
    
    # 可視化
    env.render(sleep_time=0.2)
    
    state = {i: trained_brain.preprocess_state(next_obs, i) for i in range(env.num_agents)}
    
    if done:
        print(f"テスト完了！ 最終カバレッジ: {info['coverage']*100:.1f}%")
        plt.pause(2.0)
        break
        
plt.ioff()
plt.show()