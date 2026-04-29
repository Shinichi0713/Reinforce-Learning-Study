import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from PIL import Image
import os


class RobotCarryEnv(gym.Env):
    """
    ロボットが物体を掴んで目的地まで運ぶ2Dシミュレーション環境
    gif保存機能付き
    """

    def __init__(self, max_steps=200, world_size=10.0):
        super().__init__()
        self.max_steps = max_steps
        self.world_size = world_size
        self.step_count = 0

        # 観測空間: [rx, ry, vx, vy, ox, oy, tx, ty, grasped]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        # 行動空間: [ax, ay] ロボットの速度変化量（連続）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 物理パラメータ
        self.dt = 0.1
        self.max_speed = 2.0
        self.grasp_dist = 0.5
        self.target_dist = 0.5

        # 状態変数
        self.robot_pos = None
        self.robot_vel = None
        self.object_pos = None
        self.target_pos = None
        self.grasped = None

        # 前ステップの距離（報酬計算用）
        self.prev_dist_obj = None      # ロボット〜物体の距離
        self.prev_dist_target = None   # ロボット〜目的地の距離

        # gif保存用
        self.record_frames = []  # フレーム（PIL画像）を保存するリスト
        self.recording = False   # 録画中かどうか

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # ロボット初期位置（中心付近）
        self.robot_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.robot_vel = np.array([0.0, 0.0], dtype=np.float32)

        # 物体と目的地をランダムに配置
        self.object_pos = self._random_pos()
        self.target_pos = self._random_pos()
        self.grasped = 0.0

        # 距離の初期化
        self.prev_dist_obj = np.linalg.norm(self.robot_pos - self.object_pos)
        self.prev_dist_target = np.linalg.norm(self.robot_pos - self.target_pos)

        # 録画リセット
        self.record_frames.clear()
        self.recording = False

        obs = self._get_obs()
        info = {}
        return obs, info

    def _random_pos(self):
        return self.np_random.uniform(
            low=-self.world_size / 2,
            high=self.world_size / 2,
            size=(2,)
        ).astype(np.float32)

    def _get_obs(self):
        # 生の観測
        raw_obs = np.concatenate([
            self.robot_pos,      # 0,1
            self.robot_vel,      # 2,3
            self.object_pos,     # 4,5
            self.target_pos,     # 6,7
            [self.grasped]      # 8
        ]).astype(np.float32)

        # 観測の正規化
        normalized_obs = np.zeros_like(raw_obs)
        scale_pos = self.world_size
        scale_vel = self.max_speed

        # 位置関連: 世界サイズで割る
        normalized_obs[0:2] = raw_obs[0:2] / scale_pos  # rx, ry
        normalized_obs[4:6] = raw_obs[4:6] / scale_pos  # ox, oy
        normalized_obs[6:8] = raw_obs[6:8] / scale_pos  # tx, ty

        # 速度関連: 最大速度で割る
        normalized_obs[2:4] = raw_obs[2:4] / scale_vel  # vx, vy

        # grasped はそのまま
        normalized_obs[8] = raw_obs[8]

        return normalized_obs

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. 速度変化量として扱う（加速度）
        self.robot_vel += action * self.dt  # 行動は加速度として解釈

        # 速度の大きさを制限
        speed = np.linalg.norm(self.robot_vel)
        if speed > self.max_speed:
            self.robot_vel = self.robot_vel / speed * self.max_speed

        # 2. 位置を更新
        self.robot_pos += self.robot_vel * self.dt

        # 世界の境界でクリップ
        self.robot_pos = np.clip(self.robot_pos, -self.world_size/2, self.world_size/2)

        # 2. 把持判定の修正（一度掴んだら維持）
        prev_grasped = self.grasped
        dist_to_obj = np.linalg.norm(self.robot_pos - self.object_pos)

        if self.grasped < 0.5:
            if dist_to_obj < self.grasp_dist:
                self.grasped = 1.0

        # 掴んでいる間は物体をロボットの位置に固定
        if self.grasped > 0.5:
            self.object_pos = self.robot_pos.copy()

        # 3. 報酬設計の整理
        reward = 0.0
        done = False

        if self.grasped < 0.5:
            # 掴む前：物体へ近づくインセンティブ
            reward -= 0.1 * dist_to_obj
        else:
            # 掴んだ後：目的地へ近づくインセンティブ
            dist_to_target = np.linalg.norm(self.object_pos - self.target_pos)
            reward -= 0.1 * dist_to_target

            # 目的地に到着
            if dist_to_target < self.target_dist:
                reward += 50.0  # ゴール報酬
                done = True

        # 掴んだ瞬間のボーナス（1回切り）
        if prev_grasped < 0.5 and self.grasped > 0.5:
            reward += 5.0

        # タイムアウト
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        return obs, reward, done, truncated, {}

    def render(self, mode="human"):
        if mode == "human":
            # テキスト表示
            print(f"Step {self.step_count}")
            print(f"Robot: {self.robot_pos}, Vel: {self.robot_vel}")
            print(f"Object: {self.object_pos}, Grasped: {self.grasped}")
            print(f"Target: {self.target_pos}")
            print("---")
        elif mode == "rgb_array":
            # gif保存用の画像を生成
            return self._render_rgb_array()
        else:
            super().render(mode=mode)

    def _render_rgb_array(self):
        # matplotlibで描画 → numpy配列に変換
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.world_size / 2, self.world_size / 2)
        ax.set_ylim(-self.world_size / 2, self.world_size / 2)
        ax.set_aspect("equal")
        ax.grid(True)

        # ロボット（青）
        ax.plot(self.robot_pos[0], self.robot_pos[1], "bo", markersize=10, label="Robot")

        # 物体（緑）
        color_obj = "green" if self.grasped > 0.5 else "lime"
        ax.plot(self.object_pos[0], self.object_pos[1], "o", color=color_obj, markersize=8, label="Object")

        # 目的地（赤）
        ax.plot(self.target_pos[0], self.target_pos[1], "ro", markersize=8, label="Target")

        ax.legend()
        fig.canvas.draw()

        # numpy配列に変換（matplotlibのバージョン差異に対応）
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]  # RGBA -> RGB
        plt.close(fig)
        return buf

    def start_recording(self):
        """録画を開始（stepごとにフレームを保存）"""
        self.record_frames.clear()
        self.recording = True

    def stop_recording(self):
        """録画を停止"""
        self.recording = False

    def save_gif(self, filepath, duration=100):
        """
        録画したフレームをgifとして保存
        filepath: 保存先パス（例: "episode.gif"）
        duration: フレーム間の表示時間（ms）
        """
        if not self.record_frames:
            print("No frames recorded. Call start_recording() and run steps first.")
            return

        # PIL画像リストからgifを作成
        self.record_frames[0].save(
            filepath,
            save_all=True,
            append_images=self.record_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {filepath}")

    def step_with_record(self, action):
        """録画しながらステップを進める（簡易ラッパー）"""
        obs, reward, done, truncated, info = self.step(action)
        if self.recording:
            rgb_array = self.render(mode="rgb_array")
            img = Image.fromarray(rgb_array)
            self.record_frames.append(img)
        return obs, reward, done, truncated, info