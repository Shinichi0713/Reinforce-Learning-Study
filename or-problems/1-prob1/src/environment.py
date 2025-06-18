import torch
import numpy as np
import matplotlib.pyplot as plt

# TSP環境クラス
class TSPEnv:
    def __init__(self, batch_size, num_cities=10):
        self.batch_size = batch_size
        self.num_cities = num_cities
        self.cities = None
        self.visited = None

    def reset(self):
        # 都市座標をランダム生成（[0,1]区間）
        self.cities = np.random.rand(self.batch_size, self.num_cities, 2)
        self.cities = torch.tensor(self.cities, dtype=torch.float32)
        self.visited = np.zeros((self.batch_size, self.num_cities), dtype=bool)
        self.current_city = np.random.randint(self.num_cities)
        self.visited[np.arange(self.batch_size), self.current_city] = True
        return self.cities

    # 巡回経路の長さを計算
    # 長さの負の数を報酬と定義
    def compute_tour_length(self, tour_idx):
        tour = torch.gather(self.cities.to(tour_idx.device), 1, tour_idx.unsqueeze(2).expand(-1, -1, 2))
        tour_shift = torch.roll(tour, shifts=-1, dims=1)
        length = ((tour - tour_shift) ** 2).sum(2).sqrt().sum(1)
        return length

    # エージェントが選択した順序をもとに描画
    def render(self, tour_idx):
        """
        tour_idx: (batch_size, seq_len) 各バッチごとの巡回都市順インデックス
        """
        ncols = min(4, self.batch_size)
        nrows = (self.batch_size + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)  # 1次元配列化

        for b in range(self.batch_size):
            ax = axes[b]
            cities_b = self.cities[b].cpu().numpy()  # (num_cities, 2)
            order = tour_idx[b].cpu().numpy()        # (seq_len,)
            path = cities_b[order]
            # 都市を点で描画
            ax.scatter(cities_b[:, 0], cities_b[:, 1], c='blue')
            # 巡回経路を線で描画
            ax.plot(path[:, 0], path[:, 1], c='red', marker='o')
            # スタートとゴールを強調
            ax.scatter(path[0, 0], path[0, 1], c='green', s=100, label='Start')
            ax.scatter(path[-1, 0], path[-1, 1], c='orange', s=100, label='End')
            # 始点と終点をつなぐ
            ax.plot([path[-1, 0], path[0, 0]], [path[-1, 1], path[0, 1]], 'o-', c='blue')
            ax.set_title(f'Batch {b}')
            ax.legend()

        # 不要なsubplotを消す
        for b in range(self.batch_size, nrows * ncols):
            fig.delaxes(axes[b])

        plt.tight_layout()
        plt.show()
