import numpy as np
import random
from typing import Dict, Tuple, List
# Matplotlib関連は学習コードでは不要なため省略（視覚化に必要であれば再導入）
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
import time

# (WarehouseEnv クラス定義は省略 - 元のコードをそのまま使用)

# 環境設定
GRID_SIZE = 10
NUM_AGENTS = 2
NUM_ORDERS = 3
PICKUP_LOCATIONS = [(1, 1), (8, 1), (5, 8)]
DROPOFF_LOCATION = (5, 5)

class WarehouseEnv:
    def __init__(self, size: int = GRID_SIZE, num_agents: int = NUM_AGENTS):
        self.size = size
        self.num_agents = num_agents
        self.action_space = 5  # 0:待機, 1:上, 2:下, 3:左, 4:右
        
        # Matplotlib用の図の保持
        self.fig = None
        self.ax = None
        
        # 状態の初期化
        self.reset()

    def reset(self) -> Dict[int, Tuple]:
        """環境を初期化し、初期状態を返します。"""
        self.agent_positions: Dict[int, Tuple[int, int]] = {
            i: (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            for i in range(self.num_agents)
        }
        self.agent_holding: Dict[int, bool] = {i: False for i in range(self.num_agents)}
        self.remaining_orders: List[int] = list(range(NUM_ORDERS))
        return self._get_obs()

    def _get_obs(self) -> Dict[int, Tuple]:
        obs = {}
        for i in range(self.num_agents):
            obs[i] = (
                self.agent_positions[i],
                self.agent_holding[i],
                tuple(self.remaining_orders)
            )
        return obs

    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        next_positions: Dict[int, Tuple[int, int]] = {}
        rewards: Dict[int, float] = {i: -0.1 for i in range(self.num_agents)}
        
        # 1. 位置の更新
        for i, action in actions.items():
            current_x, current_y = self.agent_positions[i]
            next_x, next_y = current_x, current_y

            if action == 1: next_y += 1  # 上
            elif action == 2: next_y -= 1  # 下
            elif action == 3: next_x -= 1  # 左
            elif action == 4: next_x += 1  # 右
            
            next_x = np.clip(next_x, 0, self.size - 1)
            next_y = np.clip(next_y, 0, self.size - 1)
            next_positions[i] = (next_x, next_y)

        # 2. 衝突判定
        final_positions = self.agent_positions.copy()
        is_collision = False
        
        for i in range(self.num_agents):
            pos = next_positions[i]
            is_colliding = False
            for j in range(self.num_agents):
                if i != j and pos == next_positions[j]:
                    is_colliding = True
                    break
            if is_colliding:
                rewards[i] -= 5.0
                is_collision = True
            else:
                final_positions[i] = pos
        
        self.agent_positions = final_positions

        # 3. ピックアップ・ドロップオフ
        for i in range(self.num_agents):
            current_pos = self.agent_positions[i]
            if not self.agent_holding[i]:
                for order_idx in self.remaining_orders:
                    if current_pos == PICKUP_LOCATIONS[order_idx]:
                        self.agent_holding[i] = True
                        self.remaining_orders.remove(order_idx)
                        rewards[i] += 10.0
                        break
            elif self.agent_holding[i]:
                if current_pos == DROPOFF_LOCATION:
                    self.agent_holding[i] = False
                    rewards[i] += 50.0
                    
        done = {i: len(self.remaining_orders) == 0 for i in range(self.num_agents)}
        return self._get_obs(), rewards, done, {"collision": is_collision}

    # --- 追加された可視化メソッド ---
    def render(self, mode='text', sleep_time=0.5):
        """
        環境を可視化します。
        mode='text': コンソールに文字で表示
        mode='graphic': Matplotlibで図として表示
        """
        if mode == 'text':
            self._render_text()
        elif mode == 'graphic':
            self._render_graphic(sleep_time)

    def _render_text(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # 場所のマーク (y座標は下から上へ増えるため、表示時は反転させるか注意が必要)
        # ここでは (0,0) を左下として扱います
        x, y = DROPOFF_LOCATION
        grid[self.size - 1 - y][x] = 'D'  # Dropoff

        for idx in self.remaining_orders:
            x, y = PICKUP_LOCATIONS[idx]
            grid[self.size - 1 - y][x] = 'P'  # Pickup

        for i, pos in self.agent_positions.items():
            x, y = pos
            char = f'A{i}'
            if self.agent_holding[i]:
                char = f'H{i}' # Holding
            
            # 同じ場所に重なった場合の表示処理（簡易）
            if grid[self.size - 1 - y][x] not in ['.', 'P', 'D']:
                grid[self.size - 1 - y][x] += char
            else:
                grid[self.size - 1 - y][x] = char

        print("-" * (self.size * 3))
        for row in grid:
            print(" ".join([f"{c:>2}" for c in row]))
        print("-" * (self.size * 3))

    def _render_graphic(self, sleep_time):
        if self.fig is None:
            plt.ion() # インタラクティブモードON
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))
        self.ax.grid(True)
        self.ax.set_title(f"Orders Remaining: {len(self.remaining_orders)}")

        # ドロップオフ地点 (赤色)
        dx, dy = DROPOFF_LOCATION
        self.ax.add_patch(patches.Rectangle((dx-0.5, dy-0.5), 1, 1, color='red', alpha=0.3, label='Dropoff'))
        self.ax.text(dx, dy, 'Drop', ha='center', va='center', fontsize=8, color='darkred')

        # ピックアップ地点 (青色)
        for idx in self.remaining_orders:
            px, py = PICKUP_LOCATIONS[idx]
            self.ax.add_patch(patches.Rectangle((px-0.5, py-0.5), 1, 1, color='blue', alpha=0.3, label='Pickup'))
            self.ax.text(px, py, 'Pick', ha='center', va='center', fontsize=8, color='darkblue')

        # エージェント (円)
        colors = ['green', 'orange', 'purple', 'cyan']
        for i, pos in self.agent_positions.items():
            ax, ay = pos
            color = colors[i % len(colors)]
            # 荷物を持っている場合は枠線を太く、色を変えるなどの表現
            edgecolor = 'black'
            linewidth = 1
            if self.agent_holding[i]:
                linewidth = 3
                edgecolor = 'red' # 荷物持ち強調
            
            circle = patches.Circle((ax, ay), 0.3, facecolor=color, edgecolor=edgecolor, linewidth=linewidth, label=f'Agent {i}')
            self.ax.add_patch(circle)
            self.ax.text(ax, ay, f'A{i}', ha='center', va='center', color='white', fontweight='bold')

        plt.draw()
        plt.pause(sleep_time)
        
# --- IQLのためのヘルパー関数 ---
def obs_to_key(obs: Tuple) -> str:
    """観測タプルをQテーブルのキーとして使用できる文字列に変換します。"""
    # 観測: (自身の位置(x,y), 荷物保持状態(bool), 未完了の注文(tuple))
    pos_str = f"P{obs[0][0]},{obs[0][1]}"
    hold_str = "H" if obs[1] else "N"
    order_str = "O" + "_".join(map(str, sorted(obs[2])))
    return f"{pos_str}_{hold_str}_{order_str}"

# --- IQL Agent クラス ---
class IQAgent:
    def __init__(self, agent_id: int, action_space_size: int, learning_rate: float = 0.1, discount_factor: float = 0.99, exploration_rate: float = 1.0, min_exploration_rate: float = 0.05, exploration_decay_rate: float = 0.9999):
        self.id = agent_id
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = min_exploration_rate
        self.epsilon_decay = exploration_decay_rate
        
        # Qテーブル: {状態キー(str): [Q(a0), Q(a1), ...]}
        self.q_table: Dict[str, np.ndarray] = {}

    def get_q_values(self, state_key: str) -> np.ndarray:
        """指定された状態のQ値を返すか、存在しない場合は初期化します。"""
        if state_key not in self.q_table:
            # Q値をランダムに初期化
            self.q_table[state_key] = np.zeros(self.action_space_size, dtype=float)
        return self.q_table[state_key]

    def choose_action(self, obs: Tuple) -> int:
        """ε-greedy戦略に基づいて行動を選択します。"""
        state_key = obs_to_key(obs)

        if random.random() < self.epsilon:
            # 探索 (Exploration): ランダムに行動
            return random.randint(0, self.action_space_size - 1)
        else:
            # 活用 (Exploitation): Q値が最大の行動
            q_values = self.get_q_values(state_key)
            return np.argmax(q_values).item()

    def learn(self, current_obs: Tuple, action: int, reward: float, next_obs: Tuple):
        """Q学習の更新ルールに従ってQテーブルを更新します。"""
        current_key = obs_to_key(current_obs)
        next_key = obs_to_key(next_obs)

        current_q = self.get_q_values(current_key)[action]
        
        # 次の状態の最大Q値 (Q(s', a'))
        max_next_q = np.max(self.get_q_values(next_key))
        
        # ターゲット値 (目標Q値)
        target_q = reward + self.gamma * max_next_q
        
        # Q値の更新: Q(s, a) = Q(s, a) + LR * (Target - Q(s, a))
        self.q_table[current_key][action] += self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """ε値を減衰させます。"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# --- 学習メインコード ---

def train_iql(env: WarehouseEnv, num_episodes: int = 10000):
    """独立型Q学習を使用してエージェントを訓練します。"""
    
    # 1. エージェントの初期化
    agents = {
        i: IQAgent(i, env.action_space)
        for i in range(env.num_agents)
    }

    # 2. 学習ループ
    total_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = {i: False for i in range(env.num_agents)}
        episode_reward = 0
        max_steps = 1000 # エピソードの最大ステップ数

        for step in range(max_steps):
            # 3. 行動選択
            actions = {i: agents[i].choose_action(obs[i]) for i in range(env.num_agents)}
            
            # 4. 環境の実行
            next_obs, rewards, done, info = env.step(actions)
            
            # 5. 学習とεの減衰
            for i in range(env.num_agents):
                agents[i].learn(obs[i], actions[i], rewards[i], next_obs[i])
                agents[i].decay_epsilon()
            
            # 報酬の集計と状態の更新
            episode_reward += sum(rewards.values())
            obs = next_obs
            
            # 終了判定 (すべてのエージェントのタスクが完了)
            if all(done.values()):
                break
        
        # ログ記録
        total_rewards.append(episode_reward)

        if (episode + 1) % 1000 == 0:
            print(f"Episode: {episode + 1}/{num_episodes}, Avg Reward (last 100): {np.mean(total_rewards[-100:]):.2f}, Epsilon: {agents[0].epsilon:.4f}")

    print("学習完了。")
    return agents, total_rewards


# --- 実行 ---
if __name__ == '__main__':
    # 視覚化のため、WarehouseEnvの定義全体が必要です
    # 実際の実行には、前の回答で提供された `WarehouseEnv` クラスをこのコードの上部にコピー＆ペーストしてください。
    
    # 環境のインスタンス化
    env = WarehouseEnv()
    
    # 学習の実行
    trained_agents, rewards_history = train_iql(env, num_episodes=50000)
    
    # 結果のプロット (学習の進捗確認)
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid')) # 100エピソードの移動平均
    plt.title('IQL Training Progress (Smoothed Total Reward)')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward')
    plt.grid(True)
    plt.show()

    # --- 学習後のテスト (10ステップ) ---
    env.reset()
    for t in range(10):
        # ε=0（活用のみ）として行動を選択
        test_actions = {i: trained_agents[i].choose_action(env._get_obs()[i]) for i in range(env.num_agents)}
        
        env.step(test_actions)
        
        # 視覚化 (Matplotlibが環境にインストールされている場合)
        print(f"\n--- Test Step {t+1} ---")
        env.render(mode='text')
        # env.render(mode='graphic', sleep_time=0.5)
        
        if all(env._get_obs()[i][2] == () for i in range(env.num_agents)):
             print("全てのタスク完了！")
             break