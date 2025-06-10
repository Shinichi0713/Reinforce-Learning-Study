import numpy as np

class TSPEnv:
    def __init__(self, num_cities=10, seed=None):
        self.num_cities = num_cities
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.cities = None
        self.visited = None
        self.current_city = None
        self.total_distance = None
        self.step_count = None

    def reset(self):
        # 都市座標をランダム生成（[0,1]区間）
        self.cities = self.rng.rand(self.num_cities, 2)
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.current_city = self.rng.randint(self.num_cities)
        self.visited[self.current_city] = True
        self.total_distance = 0.0
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        # 状態: [都市座標, 訪問フラグ, 現在地]
        return {
            'cities': self.cities.copy(),  # (num_cities, 2)
            'visited': self.visited.copy(),  # (num_cities,)
            'current_city': self.current_city
        }

    def step(self, action):
        # action: 次に移動する都市のインデックス
        if self.visited[action]:
            # 既に訪問済み都市を選んだ場合は大きなペナルティ
            reward = -100.0
            done = True
            return self._get_obs(), reward, done, {}

        # 距離計算
        prev_city = self.current_city
        dist = np.linalg.norm(self.cities[prev_city] - self.cities[action])
        self.total_distance += dist
        self.current_city = action
        self.visited[action] = True
        self.step_count += 1

        done = self.step_count == self.num_cities - 1
        reward = -dist  # 距離のマイナスを報酬

        if done:
            # 最後はスタート地点に戻る
            start_city = np.where(self.visited)[0][0]
            dist_return = np.linalg.norm(self.cities[self.current_city] - self.cities[start_city])
            self.total_distance += dist_return
            reward -= dist_return

        return self._get_obs(), reward, done, {}

    def render(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.cities[:,0], self.cities[:,1], c='blue')
        order = [np.where(self.visited)[0][0]]  # スタート地点
        for i, v in enumerate(self.visited):
            if v and i not in order:
                order.append(i)
        order.append(order[0])  # return to start
        path = self.cities[order]
        plt.plot(path[:,0], path[:,1], c='red')
        plt.title(f'Total distance: {self.total_distance:.2f}')
        plt.show()

# 使い方例
if __name__ == '__main__':
    env = TSPEnv(num_cities=5, seed=42)
    obs = env.reset()
    print('都市座標:', obs['cities'])
    for _ in range(4):
        # ランダムに未訪問都市を選択
        unvisited = np.where(~obs['visited'])[0]
        action = np.random.choice(unvisited)
        obs, reward, done, _ = env.step(action)
        print(f'都市{action}へ移動, 報酬: {reward:.2f}')
        if done:
            break
    env.render()
