import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from PIL import Image
import os


class RobotCarryEnv(gym.Env):
    """
    ロボットが物体を掴んで目的地まで運ぶ2Dシミュレーション環境
    行動は「1ステップでの移動量（dx, dy）」として扱い、1ステップで1マス動くように修正
    gif保存機能付き
    """

    def __init__(self, max_steps=200, world_size=10.0):
        super().__init__()
        self.max_steps = max_steps
        self.world_size = world_size
        self.step_count = 0

        # 観測空間: 15次元
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # 行動空間: [dx, dy] ロボットの1ステップでの移動量（連続）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 物理パラメータ
        self.dt = 1.0  # 1ステップで1マス動くように固定
        self.max_speed = 1.0  # 1ステップで動ける最大距離（行動空間の上限と同じ）
        self.grasp_dist = 0.5
        self.target_dist = 0.5

        # 状態変数
        self.robot_pos = None
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
            [0.0, 0.0],          # 2,3（速度は使わないが次元を維持）
            self.object_pos,     # 4,5
            self.target_pos,     # 6,7
            [self.grasped]      # 8
        ]).ast(np.float32)

        # 相対位置と距離
        rel_obj = self.object_pos - self.robot_pos
        rel_target = self.target_pos - self.robot_pos
        dist_obj = np.linalg.norm(rel_obj)
        dist_target = np.linalg.norm(rel_target)

        # 拡張観測
        extended_obs = np.concatenate([
            raw_obs,
            rel_obj,           # 9,10
            rel_target,        # 11,12
            [dist_obj],        # 13
            [dist_target]      # 14
        ]).astype(np.float32)

        # 正規化
        scale_pos = self.world_size
        normalized_obs = extended_obs.copy()
        normalized_obs[0:2] /= scale_pos    # rx, ry
        normalized_obs[4:6] /= scale_pos    # ox, oy
        normalized_obs[6:8] /= scale_pos    # tx, ty
        normalized_obs[9:11] /= scale_pos   # rel_obj
        normalized_obs[11:13] /= scale_pos  # rel_target
        normalized_obs[13] /= scale_pos     # dist_obj
        normalized_obs[14] /= scale_pos     # dist_target
        # 速度は使わないので正規化しない（0のまま）

        return normalized_obs

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. 行動を「1ステップでの移動量」として扱う
        # dt=1.0 なので robot_pos += action * dt = action
        move = action * self.dt

        # 移動量の大きさを制限（1ステップで動ける最大距離）
        move_norm = np.linalg.norm(move)
        if move_norm > self.max_speed:
            move = move / move_norm * self.max_speed

        # 2. 位置を更新
        self.robot_pos += move

        # 世界の境界でクリップ
        self.robot_pos = np.clip(self.robot_pos, -self.world_size/2, self.world_size/2)

        # 3. 把持判定（一度掴んだら維持）
        prev_grasped = self.grasped
        dist_to_obj = np.linalg.norm(self.robot_pos - self.object_pos)

        if self.grasped < 0.5:
            if dist_to_obj < self.grasp_dist:
                self.grasped = 1.0

        # 掴んでいる間は物体をロボットの位置に固定
        if self.grasped > 0.5:
            self.object_pos = self.robot_pos.copy()

        # 4. 報酬設計
        reward = 0.0
        done = False
        dist_to_target = np.linalg.norm(self.object_pos - self.target_pos)
        # 距離を正規化
        norm_dist_obj = dist_to_obj / self.world_size
        norm_dist_target = dist_to_target / self.world_size

        if self.grasped < 0.5:
            reward -= 0.1 * norm_dist_obj  # 正規化された距離に基づく報酬
        else:
            reward -= 0.1 * norm_dist_target
            if dist_to_target < self.target_dist:
                reward += 10.0
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
            print(f"Robot: {self.robot_pos}")
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
    


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    import time

    # 自作環境をインポート（ファイル名に応じて変更してください）
    # from robot_carry_env import RobotCarryEnv
    # または、このノートブック内で既に定義されている場合はそのまま使う

    # 環境の作成
    env = RobotCarryEnv(max_steps=200, world_size=10.0)

    # 1エピソード分のランダム行動で動作確認
    obs, info = env.reset()
    env.start_recording()  # 録画開始

    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        # ランダム行動（連続行動空間）
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step_with_record(action)
        total_reward += reward

    env.stop_recording()  # 録画停止

    print(f"Episode finished. Total reward: {total_reward:.2f}")
    print(f"Final step: {env.step_count}, Grasped: {env.grasped}")

    # gif として保存
    env.save_gif("random_agent_episode.gif", duration=100)
